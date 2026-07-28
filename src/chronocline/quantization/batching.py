"""Timestamp batching transformations."""

from __future__ import annotations

import numpy as np


def batch_timestamps(
    times: np.ndarray, window: float, phase: float = 0.0, *, ceiling: bool = False
) -> np.ndarray:
    """Quantize timestamps to fixed start or end batching windows."""
    if window <= 0:
        raise ValueError("window must be positive")
    scaled = (np.asarray(times, dtype=float) - phase) / window
    rule = np.ceil if ceiling else np.floor
    return phase + window * rule(scaled)
