"""Exact CDF-difference construction of quantized memoryless timing channels."""

from __future__ import annotations

import numpy as np

from ..distributions.base import JitterDistribution
from ..exceptions import UnsupportedScientificModelError
from ..quantization import UniformQuantizer
from .matrix import ChannelMatrix


def stable_interval_probability(
    jitter: JitterDistribution, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    """Return ``P(lower <= Z < upper)`` without subtracting saturated CDF values.

    The lower-tail representation is stable near negative infinity; the survival
    representation is stable in the far upper tail.  A material disagreement is
    treated as a numerical error rather than silently clipped into probability.
    """
    lower_cdf = jitter.cdf(lower)
    upper_cdf = jitter.cdf(upper)
    cdf_difference = upper_cdf - lower_cdf
    survival_difference = jitter.sf(lower) - jitter.sf(upper)
    use_survival = np.asarray(lower, float) >= jitter.mean()
    chosen = np.where(use_survival, survival_difference, cdf_difference)
    scale = np.maximum(np.abs(cdf_difference), np.abs(survival_difference))
    tolerance = 5e-15 + 1e-8 * scale
    inconsistent = (
        (cdf_difference > 0)
        & (survival_difference > 0)
        & (np.abs(cdf_difference - survival_difference) > tolerance)
    )
    if np.any(inconsistent):
        raise RuntimeError("CDF and survival interval calculations materially disagree")
    if np.any(chosen < -1e-14):
        raise RuntimeError("stable interval calculation produced negative probability")
    return np.maximum(chosen, 0.0), {
        "cdf_difference": int(np.size(use_survival) - np.count_nonzero(use_survival)),
        "survival_difference": int(np.count_nonzero(use_survival)),
    }


def _index_limits(
    alphabet: np.ndarray, jitter: JitterDistribution, quantizer: UniformQuantizer, tail: float
) -> tuple[int, int]:
    lo = float(np.min(alphabet) + jitter.ppf(tail / 2))
    hi = float(np.max(alphabet) + jitter.ppf(1 - tail / 2))
    offset = 0.5 if quantizer.mode == "nearest" else 0.0
    return int(np.floor((lo - quantizer.phase) / quantizer.step - offset)), int(
        np.floor((hi - quantizer.phase) / quantizer.step + offset)
    )


def build_memoryless_channel(
    alphabet: np.ndarray | list[float],
    jitter: JitterDistribution,
    quantizer: UniformQuantizer,
    *,
    tail_probability: float = 1e-12,
    include_overflow_bins: bool = True,
    random_phase_mode: str | bool = False,
) -> ChannelMatrix:
    """Build a shared-support exact channel using jitter CDF bin differences.

    Finite interior bins have exact probability under the supplied CDF; optional
    first and last columns retain all omitted lower and upper tail mass.
    """
    inputs = np.asarray(alphabet, dtype=float)
    if inputs.ndim != 1 or len(inputs) == 0 or np.any(np.diff(inputs) <= 0):
        raise ValueError("alphabet must be a non-empty strictly ordered vector")
    if random_phase_mode in {"per_trace", "per_symbol_unknown"}:
        raise NotImplementedError(
            f"{random_phase_mode} phase is not a memoryless DMC; use an explicit stateful model"
        )
    supported_phase_modes = {
        False,
        "fixed_known",
        "fixed_unknown",
        "per_symbol_receiver_known",
    }
    if random_phase_mode not in supported_phase_modes:
        raise ValueError(f"unknown random phase mode {random_phase_mode}")
    configured_metadata = quantizer.metadata()
    start, end = _index_limits(inputs, jitter, quantizer, tail_probability)
    indexes = np.arange(start, end + 1)
    centers = quantizer.phase + indexes * quantizer.step
    if quantizer.mode == "nearest":
        lower, upper = centers - quantizer.step / 2, centers + quantizer.step / 2
    else:
        lower, upper = centers, centers + quantizer.step
    interior, interval_diagnostics = stable_interval_probability(
        jitter,
        lower[None, :] - inputs[:, None],
        upper[None, :] - inputs[:, None],
    )
    omitted_mass = 1 - interior.sum(axis=1)
    if include_overflow_bins:
        low_tail = jitter.cdf(lower[0] - inputs)[:, None]
        high_tail = jitter.sf(upper[-1] - inputs)[:, None]
        matrix = np.concatenate([low_tail, interior, high_tail], axis=1)
        outputs = np.concatenate(
            [
                np.array(["lower_overflow"], dtype=object),
                indexes.astype(object),
                np.array(["upper_overflow"], dtype=object),
            ]
        )
    else:
        retained = interior.sum(axis=1, keepdims=True)
        if np.any(retained <= 0):
            raise RuntimeError("conditional truncation retained zero probability mass")
        matrix = interior / retained
        outputs = indexes.astype(object)
    residual = 1 - matrix.sum(axis=1)
    if np.max(np.abs(residual)) > 1e-10:
        raise RuntimeError("CDF calculation did not preserve channel mass")
    if include_overflow_bins:
        matrix[:, -1] += residual
    return ChannelMatrix(
        inputs,
        outputs,
        matrix,
        {
            "tail_probability": tail_probability,
            "tail_mode": "overflow_bins" if include_overflow_bins else "conditional_truncation",
            "omitted_mass_per_row": omitted_mass.tolist(),
            "transition_probability_methods": interval_diagnostics,
            "upper_tail_method": "survival_function",
            "quantizer": configured_metadata,
            "construction": "exact_cdf_difference",
        },
    )


def build_random_phase_channel(
    alphabet: np.ndarray | list[float],
    jitter: JitterDistribution,
    step: float,
    *,
    points: int = 32,
    tail_probability: float = 1e-12,
) -> ChannelMatrix:
    """Reject the deprecated ambiguous phase-averaged channel construction."""
    del alphabet, jitter, step, points, tail_probability
    raise UnsupportedScientificModelError(
        "phase averaging has no unique receiver semantics; use an explicit phase mode"
    )


def monte_carlo_matrix(
    channel: ChannelMatrix,
    jitter: JitterDistribution,
    quantizer: UniformQuantizer,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Independently estimate fixed-quantizer matrix rows for validation only."""
    if samples < 1:
        raise ValueError("samples must be positive")
    estimates = np.zeros_like(channel.probabilities)
    labels = {label: j for j, label in enumerate(channel.outputs)}
    interior = [int(x) for x in channel.outputs if isinstance(x, (int, np.integer))]
    lo, hi = min(interior), max(interior)
    for i, symbol in enumerate(channel.inputs):
        indexes = quantizer.bin_index(symbol + jitter.sample(samples, rng))
        has_overflow = "lower_overflow" in labels and "upper_overflow" in labels
        if has_overflow:
            estimates[i, labels["lower_overflow"]] = np.mean(indexes < lo)
            estimates[i, labels["upper_overflow"]] = np.mean(indexes > hi)
        else:
            retained = indexes[(indexes >= lo) & (indexes <= hi)]
            while len(retained) < samples:
                additional = quantizer.bin_index(symbol + jitter.sample(samples, rng))
                retained = np.r_[retained, additional[(additional >= lo) & (additional <= hi)]]
            indexes = retained[:samples]
        for index in range(lo, hi + 1):
            estimates[i, labels[index]] = np.mean(indexes == index)
    return estimates
