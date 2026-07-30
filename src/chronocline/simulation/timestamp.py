"""Explicit stateful timing-observation stages and packet identity."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..quantization import UniformQuantizer


@dataclass(frozen=True)
class ObservationTrace:
    """One aligned ``X -> S -> A -> R -> Y`` observation trace.

    ``packet_ids`` remain attached to each row even when arrival ordering is
    preserved; a batching transformation only changes release/observation fields.
    """

    intended_delays: np.ndarray
    ideal_timestamps: np.ndarray
    arrival_timestamps: np.ndarray
    observed_timestamps: np.ndarray
    observed_delays: np.ndarray
    packet_ids: np.ndarray
    batch_ids: np.ndarray
    closure_causes: np.ndarray
    model: str

    def with_batching(
        self, timestamps: np.ndarray, batch_ids: np.ndarray, causes: np.ndarray, model: str
    ) -> ObservationTrace:
        timestamps = np.asarray(timestamps, dtype=float)
        return replace(
            self,
            observed_timestamps=timestamps,
            observed_delays=np.diff(np.r_[0.0, timestamps]),
            batch_ids=np.asarray(batch_ids, dtype=int),
            closure_causes=np.asarray(causes, dtype=object),
            model=model,
        )


def observation_trace(
    delays: np.ndarray,
    jitter: np.ndarray,
    quantizer: UniformQuantizer,
    *,
    model: str,
    preserve_order: bool = True,
) -> ObservationTrace:
    """Construct a documented timing model without conflating delay/timestamp noise.

    Negative delay-jitter observations use the explicit physical ``clip_zero``
    policy. Timestamp jitter is applied to cumulative timestamps and may be made
    nondecreasing with ``preserve_order``.
    """
    intended = np.asarray(delays, dtype=float)
    noise = np.asarray(jitter, dtype=float)
    if intended.ndim != 1 or len(intended) == 0 or np.any(intended < 0):
        raise ValueError("delays must be a non-empty non-negative vector")
    if len(noise) < len(intended):
        raise ValueError("jitter must provide one value per intended delay")
    ideal = np.cumsum(intended)
    packet_ids = np.arange(len(intended), dtype=int)
    if model == "ideal_delays":
        arrivals = ideal.copy()
        observed = intended.copy()
        timestamps = ideal.copy()
    elif model == "delay_jitter":
        arrivals = ideal.copy()
        observed = np.maximum(intended + noise[: len(intended)], 0.0)
        timestamps = np.cumsum(observed)
    elif model == "delay_quantization":
        arrivals = ideal.copy()
        observed = quantizer.quantize(np.maximum(intended + noise[: len(intended)], 0.0))
        timestamps = np.cumsum(observed)
    elif model in {
        "timestamp_quantization",
        "timestamp_quantization_then_batching",
        "arrival_timestamps",
    }:
        arrivals = ideal + noise[: len(intended)]
        if preserve_order:
            arrivals = np.maximum.accumulate(arrivals)
        timestamps = arrivals if model == "arrival_timestamps" else quantizer.quantize(arrivals)
        observed = np.diff(np.r_[0.0, timestamps])
    else:
        raise ValueError(f"unsupported observation model {model}")
    return ObservationTrace(
        intended,
        ideal,
        arrivals,
        timestamps,
        observed,
        packet_ids,
        packet_ids.copy(),
        np.full(len(intended), "packet", dtype=object),
        model,
    )


def cumulative_timestamp_observations(
    delays: np.ndarray,
    jitter: np.ndarray,
    quantizer: UniformQuantizer,
    *,
    preserve_order: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible timestamp quantization view of :func:`observation_trace`."""
    trace = observation_trace(
        delays,
        jitter,
        quantizer,
        model="timestamp_quantization",
        preserve_order=preserve_order,
    )
    return trace.observed_timestamps, trace.observed_delays
