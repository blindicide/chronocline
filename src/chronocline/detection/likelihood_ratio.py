"""Optimal simple-hypothesis likelihood-ratio detection."""

from __future__ import annotations

import numpy as np


def log_likelihood_ratio(samples: np.ndarray, active: np.ndarray, baseline: np.ndarray) -> float:
    """Return sum log likelihood ratio, with rigorous infinite-support handling."""
    p1, p0 = np.asarray(active, float), np.asarray(baseline, float)
    indexes = np.asarray(samples, int)
    if np.any(p1[indexes] == 0):
        return float("-inf")
    if np.any(p0[indexes] == 0):
        return float("inf")
    return float(np.log(p1[indexes] / p0[indexes]).sum())
