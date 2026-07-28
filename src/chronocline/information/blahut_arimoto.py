"""Blahut-Arimoto capacity solver with upper/lower gap termination."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CapacityResult:
    """Auditable finite-channel capacity result."""

    capacity_bits: float
    input_probabilities: np.ndarray
    iterations: int
    converged: bool
    residual: float
    history: tuple[float, ...] = ()
    warnings: tuple[str, ...] = ()


def blahut_arimoto(
    channel: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
    initial: np.ndarray | None = None,
    history: bool = False,
) -> CapacityResult:
    """Maximize mutual information for a finite row-stochastic channel in bits."""
    w = np.asarray(channel, dtype=float)
    if w.ndim != 2 or np.any(w < -1e-12) or not np.allclose(w.sum(axis=1), 1, atol=1e-12):
        raise ValueError("channel must be a row-stochastic matrix")
    m = w.shape[0]
    p = np.full(m, 1 / m) if initial is None else np.asarray(initial, dtype=float).copy()
    p /= p.sum()
    trace: list[float] = []
    residual = float("inf")
    for _iteration in range(1, max_iterations + 1):
        q = p @ w
        mask = (w > 0) & (q[None, :] > 0)
        ratio = np.ones_like(w)
        ratio[mask] = w[mask] / np.broadcast_to(q, w.shape)[mask]
        d = np.sum(np.where(mask, w * np.log2(ratio), 0.0), axis=1)
        lower = float(p @ d)
        upper = float(np.max(d))
        residual = upper - lower
        trace.append(lower)
        if residual <= tolerance:
            break
        exponent = d - np.max(d)
        p = p * np.exp2(exponent)
        p /= p.sum()
    q = p @ w
    ratio = np.ones_like(w)
    mask = (w > 0) & (q[None, :] > 0)
    ratio[mask] = w[mask] / np.broadcast_to(q, w.shape)[mask]
    capacity = float(p @ np.sum(np.where(mask, w * np.log2(ratio), 0.0), axis=1))
    warnings = ("indistinguishable input rows",) if np.allclose(w, w[0]) else ()
    return CapacityResult(
        capacity,
        p,
        _iteration,
        residual <= tolerance,
        residual,
        tuple(trace) if history else (),
        warnings,
    )
