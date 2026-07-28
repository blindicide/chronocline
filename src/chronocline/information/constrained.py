"""Fixed-alphabet capacity optimization with auditable feasibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

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
    max_kl: float | None = None,
    max_mean_delay: float | None = None,
    tolerance: float = 1e-8,
    starts: int = 12,
    seed: int = 0,
) -> ConstrainedCapacityResult:
    """Maximize I(X;Y) subject to output KL and mean-delay limits using SLSQP."""
    w, d, p0 = np.asarray(channel, float), np.asarray(alphabet, float), np.asarray(baseline, float)
    if w.shape[0] != len(d) or w.shape[1] != len(p0):
        raise ValueError("channel, alphabet, and baseline dimensions are inconsistent")
    rng = np.random.default_rng(seed)
    initial_points = [
        np.full(len(d), 1 / len(d)),
        *np.eye(len(d)),
        *rng.dirichlet(np.ones(len(d)), size=max(0, starts - len(d) - 1)),
    ]
    constraints = [{"type": "eq", "fun": lambda p: p.sum() - 1}]
    if max_kl is not None:
        constraints.append({"type": "ineq", "fun": lambda p: max_kl - kl_divergence(p @ w, p0)})
    if max_mean_delay is not None:
        constraints.append({"type": "ineq", "fun": lambda p: max_mean_delay - float(p @ d)})
    best: tuple[float, np.ndarray, Any] | None = None
    for p in initial_points:
        result = minimize(
            lambda x: -mutual_information(x, w),
            p,
            method="SLSQP",
            bounds=[(0, 1)] * len(d),
            constraints=constraints,
            options={"ftol": tolerance, "maxiter": 1000},
        )
        candidate = np.maximum(result.x, 0)
        candidate /= candidate.sum()
        output = candidate @ w
        feasible = (max_kl is None or kl_divergence(output, p0) <= max_kl + tolerance) and (
            max_mean_delay is None or candidate @ d <= max_mean_delay + tolerance
        )
        if feasible and (best is None or -result.fun > best[0]):
            best = (-result.fun, candidate, result)
    if best is None:
        raise ValueError("constraints are infeasible for all tested initializations")
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
