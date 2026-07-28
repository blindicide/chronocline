"""Stable discrete entropy functions in bits."""

from __future__ import annotations

import numpy as np


def validate_distribution(probabilities: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    """Validate and lightly clip only floating-point-scale negative probabilities."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or np.any(p < -tolerance) or not np.isclose(p.sum(), 1.0, atol=tolerance):
        raise ValueError("probabilities must be non-negative and sum to one")
    return np.maximum(p, 0.0)


def entropy(probabilities: np.ndarray) -> float:
    """Return base-2 Shannon entropy, defining 0 log2(0) as zero."""
    p = validate_distribution(probabilities)
    positive = p > 0
    return float(-np.sum(p[positive] * np.log2(p[positive])))
