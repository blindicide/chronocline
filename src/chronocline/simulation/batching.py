"""Stateful batching transformations with explicit packet-to-batch mappings."""

from __future__ import annotations

import numpy as np

from .timestamp import ObservationTrace


def batch_observation_trace(
    trace: ObservationTrace,
    window: float,
    phase: float = 0.0,
    *,
    ceiling: bool = False,
    maximum_batch_size: int | None = None,
) -> ObservationTrace:
    """Apply a fixed-window release policy while retaining every packet identifier."""
    if window <= 0:
        raise ValueError("window must be positive")
    if maximum_batch_size is not None and maximum_batch_size < 1:
        raise ValueError("maximum_batch_size must be positive")
    times = trace.observed_timestamps
    slots = np.floor((times - phase) / window).astype(int)
    ids = np.empty(len(times), dtype=int)
    causes = np.empty(len(times), dtype=object)
    releases = np.empty(len(times), dtype=float)
    batch = 0
    in_batch = 0
    previous_slot: int | None = None
    for index, slot in enumerate(slots):
        slot_changed = previous_slot is not None and slot != previous_slot
        if slot_changed or (maximum_batch_size is not None and in_batch >= maximum_batch_size):
            batch += 1
            in_batch = 0
        ids[index] = batch
        in_batch += 1
        releases[index] = phase + window * (slot + int(ceiling))
        causes[index] = "maximum_batch_size" if maximum_batch_size == in_batch else "time_window"
        previous_slot = int(slot)
    if len(causes):
        causes[-1] = "trace_end" if causes[-1] == "time_window" else causes[-1]
    model = "ceiling_release" if ceiling else "fixed_window_observation"
    return trace.with_batching(releases, ids, causes, model)


def fixed_window(timestamps: np.ndarray, window: float, phase: float = 0.0) -> np.ndarray:
    """Compatibility wrapper returning start-of-window releases."""
    return phase + window * np.floor((np.asarray(timestamps) - phase) / window)


def ceiling_release(timestamps: np.ndarray, window: float, phase: float = 0.0) -> np.ndarray:
    """Compatibility wrapper returning end-of-window releases."""
    return phase + window * np.ceil((np.asarray(timestamps) - phase) / window)
