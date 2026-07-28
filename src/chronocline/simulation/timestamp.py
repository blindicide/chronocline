"""Cumulative-timestamp timing models with explicit ordering policy."""

from __future__ import annotations

import numpy as np

from ..quantization import UniformQuantizer


def cumulative_timestamp_observations(
    delays: np.ndarray,
    jitter: np.ndarray,
    quantizer: UniformQuantizer,
    *,
    preserve_order: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return quantized timestamps and their observed intervals.

    Jitter is applied to cumulative timestamps, intentionally creating output
    memory. ``preserve_order`` uses a cumulative maximum to avoid reordering.
    """
    delays, jitter = np.asarray(delays, float), np.asarray(jitter, float)
    if len(jitter) != len(delays) + 1 or np.any(delays < 0):
        raise ValueError("timestamp jitter must contain n + 1 values for n non-negative delays")
    arrivals = np.r_[0.0, np.cumsum(delays)] + jitter
    if preserve_order:
        arrivals = np.maximum.accumulate(arrivals)
    timestamps = quantizer.quantize(arrivals)
    return timestamps[1:], np.diff(timestamps)
