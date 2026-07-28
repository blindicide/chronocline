"""Distribution-level detectability measures."""

from __future__ import annotations

import numpy as np

from .entropy import validate_distribution


def kl_divergence(p: np.ndarray, q: np.ndarray, *, base: float = 2.0) -> float:
    """Return KL(P||Q), including infinity when P has mass outside Q support."""
    p, q = validate_distribution(p), validate_distribution(q)
    if np.any((p > 0) & (q == 0)):
        return float("inf")
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask]) / np.log(base)))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Return total variation distance."""
    return float(0.5 * np.abs(validate_distribution(p) - validate_distribution(q)).sum())


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Return Jensen-Shannon divergence in bits."""
    p, q = validate_distribution(p), validate_distribution(q)
    return 0.5 * kl_divergence(p, (p + q) / 2) + 0.5 * kl_divergence(q, (p + q) / 2)


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """Return Hellinger distance."""
    return float(
        np.linalg.norm(np.sqrt(validate_distribution(p)) - np.sqrt(validate_distribution(q)))
        / np.sqrt(2)
    )


def bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    """Return Bhattacharyya coefficient."""
    return float(np.sqrt(validate_distribution(p) * validate_distribution(q)).sum())


def chi_square(p: np.ndarray, q: np.ndarray) -> float:
    """Return chi-square divergence or infinity when undefined."""
    p, q = validate_distribution(p), validate_distribution(q)
    if np.any((p > 0) & (q == 0)):
        return float("inf")
    return float(np.sum(np.divide((p - q) ** 2, q, out=np.zeros_like(p), where=q > 0)))


def pinsker_upper_bound(kl_bits: float) -> float:
    """Return Pinsker TV bound after converting bits to nats."""
    return float(np.sqrt(np.log(2) * kl_bits / 2))
