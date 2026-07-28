"""Fixed-alphabet capacity optimization with auditable feasibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .blahut_arimoto import blahut_arimoto
from .divergence import kl_divergence
from .mutual_information import mutual_information


@dataclass(frozen=True)
class ConstrainedCapacityResult:
    """Constrained fixed-channel optimization result."""

    capacity_bits: float
    input_probabilities: np.ndarray
    output_probabilities: np.ndarray
    kl_divergence_bits: float
    mean_delay: float
    feasible: bool
    converged: bool
    iterations: int
    initialization: str
    warnings: tuple[str, ...] = ()


def constrained_capacity(
    channel: np.ndarray,
    alphabet: np.ndarray,
    baseline: np.ndarray,
    *,
    max_kl_bits: float | None = None,
    max_mean_delay: float | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
    starts: int = 12,
    seed: int = 0,
) -> ConstrainedCapacityResult:
    """Maximize I(X;Y) subject to output KL and mean-delay limits using SLSQP."""
    w, d, p0 = np.asarray(channel, float), np.asarray(alphabet, float), np.asarray(baseline, float)
    if w.shape[0] != len(d) or w.shape[1] != len(p0):
        raise ValueError("channel, alphabet, and baseline dimensions are inconsistent")
    rng = np.random.default_rng(seed)
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if len(d) == 2:
        grid = np.linspace(0.0, 1.0, 10_001)
        grid_feasible: list[tuple[float, np.ndarray]] = []
        for first_probability in grid:
            candidate = np.array([first_probability, 1 - first_probability])
            output = candidate @ w
            if (
                max_kl_bits is not None
                and kl_divergence(output, p0) > max_kl_bits + tolerance
            ):
                continue
            if max_mean_delay is not None and candidate @ d > max_mean_delay + tolerance:
                continue
            grid_feasible.append((mutual_information(candidate, w), candidate))
        if not grid_feasible:
            raise ValueError("constraints are infeasible for every binary grid point")
        value, candidate = max(grid_feasible, key=lambda item: item[0])
        output = candidate @ w
        return ConstrainedCapacityResult(
            float(value),
            candidate,
            output,
            kl_divergence(output, p0),
            float(candidate @ d),
            True,
            True,
            len(grid),
            "dense_binary_grid",
        )
    unconstrained = blahut_arimoto(w, tolerance=tolerance).input_probabilities
    baseline_like = np.linalg.lstsq(w.T, p0, rcond=None)[0]
    baseline_like = np.maximum(baseline_like, 0)
    baseline_like /= baseline_like.sum() if baseline_like.sum() else 1
    initial_points = [
        np.full(len(d), 1 / len(d)),
        *np.eye(len(d)),
        unconstrained,
        baseline_like,
        *rng.dirichlet(np.ones(len(d)), size=max(0, starts - len(d) - 1)),
    ]
    constraints = [{"type": "eq", "fun": lambda p: p.sum() - 1}]
    if max_kl_bits is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda p: max_kl_bits - kl_divergence(p @ w, p0)}
        )
    if max_mean_delay is not None:
        constraints.append({"type": "ineq", "fun": lambda p: max_mean_delay - float(p @ d)})
    best: tuple[float, np.ndarray, Any] | None = None
    fallback: tuple[float, np.ndarray, Any] | None = None
    for p in initial_points:
        result = minimize(
            lambda x: -mutual_information(x, w),
            p,
            method="SLSQP",
            bounds=[(0, 1)] * len(d),
            constraints=constraints,
            options={"ftol": tolerance, "maxiter": max_iterations},
        )
        candidate = np.maximum(result.x, 0)
        candidate /= candidate.sum()
        output = candidate @ w
        capacity = mutual_information(candidate, w)
        feasible = (
            max_kl_bits is None or kl_divergence(output, p0) <= max_kl_bits + tolerance
        ) and (max_mean_delay is None or candidate @ d <= max_mean_delay + tolerance)
        if feasible and (fallback is None or capacity > fallback[0]):
            fallback = (capacity, candidate, result)
        if feasible and result.success and (best is None or capacity > best[0]):
            best = (capacity, candidate, result)
    if best is None:
        if fallback is None:
            raise ValueError("constraints are infeasible for all tested initializations")
        best = fallback
    value, p, result = best
    q = p @ w
    return ConstrainedCapacityResult(
        float(value),
        p,
        q,
        kl_divergence(q, p0),
        float(p @ d),
        True,
        bool(result.success),
        int(result.nit),
        "multi_start_slsqp",
        () if result.success else (str(result.message),),
    )
