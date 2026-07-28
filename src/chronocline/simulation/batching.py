"""Batching wrappers preserving release semantics metadata."""

from __future__ import annotations

import numpy as np

from ..quantization.batching import batch_timestamps


def fixed_window(timestamps: np.ndarray, window: float, phase: float = 0.0) -> np.ndarray:
    """Release timestamps at each window start."""
    return batch_timestamps(timestamps, window, phase, ceiling=False)


def ceiling_release(timestamps: np.ndarray, window: float, phase: float = 0.0) -> np.ndarray:
    """Release timestamps at the end of the current window."""
    return batch_timestamps(timestamps, window, phase, ceiling=True)
