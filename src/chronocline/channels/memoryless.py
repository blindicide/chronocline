"""Exact CDF-difference construction of quantized memoryless timing channels."""

from __future__ import annotations

import numpy as np

from ..distributions.base import JitterDistribution
from ..quantization import UniformQuantizer
from ..quantization.random_phase import phase_nodes
from .matrix import ChannelMatrix


def _index_limits(
    alphabet: np.ndarray, jitter: JitterDistribution, quantizer: UniformQuantizer, tail: float
) -> tuple[int, int]:
    lo = float(np.min(alphabet) + jitter.ppf(tail / 2))
    hi = float(np.max(alphabet) + jitter.ppf(1 - tail / 2))
    return int(np.floor((lo - quantizer.phase) / quantizer.step)), int(
        np.floor((hi - quantizer.phase) / quantizer.step)
    )


def build_memoryless_channel(
    alphabet: np.ndarray | list[float],
    jitter: JitterDistribution,
    quantizer: UniformQuantizer,
    *,
    tail_probability: float = 1e-12,
    include_overflow_bins: bool = True,
) -> ChannelMatrix:
    """Build a shared-support exact channel using jitter CDF bin differences.

    Finite interior bins have exact probability under the supplied CDF; optional
    first and last columns retain all omitted lower and upper tail mass.
    """
    inputs = np.asarray(alphabet, dtype=float)
    if inputs.ndim != 1 or len(inputs) == 0 or np.any(np.diff(inputs) <= 0):
        raise ValueError("alphabet must be a non-empty strictly ordered vector")
    if quantizer.mode != "floor":
        # Nearest bins have the same CDF intervals after a half-step phase shift.
        quantizer = UniformQuantizer(quantizer.step, quantizer.phase - quantizer.step / 2, "floor")
    start, end = _index_limits(inputs, jitter, quantizer, tail_probability)
    indexes = np.arange(start, end + 1)
    lower = quantizer.phase + indexes * quantizer.step
    upper = lower + quantizer.step
    interior = jitter.cdf(upper[None, :] - inputs[:, None]) - jitter.cdf(
        lower[None, :] - inputs[:, None]
    )
    interior = np.maximum(interior, 0.0)
    if include_overflow_bins:
        low_tail = jitter.cdf(lower[0] - inputs)[:, None]
        high_tail = (1 - jitter.cdf(upper[-1] - inputs))[:, None]
        matrix = np.concatenate([low_tail, interior, high_tail], axis=1)
        outputs = np.concatenate(
            [
                np.array(["lower_overflow"], dtype=object),
                indexes.astype(object),
                np.array(["upper_overflow"], dtype=object),
            ]
        )
    else:
        matrix = interior
        outputs = indexes.astype(object)
    residual = 1 - matrix.sum(axis=1)
    if np.max(np.abs(residual)) > 1e-10:
        raise RuntimeError("CDF calculation did not preserve channel mass")
    matrix[:, -1] += residual
    return ChannelMatrix(
        inputs,
        outputs,
        matrix,
        {
            "tail_probability": tail_probability,
            "overflow_bins": include_overflow_bins,
            "quantizer": quantizer.metadata(),
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
    """Average fixed-phase matrices by deterministic Gauss-Legendre quadrature."""
    nodes, weights = phase_nodes(step, points)
    matrices = [
        build_memoryless_channel(
            alphabet, jitter, UniformQuantizer(step, float(phi)), tail_probability=tail_probability
        )
        for phi in nodes
    ]
    # A phase-dependent support cannot be averaged by labels safely; use a common index support.
    all_outputs = sorted(
        {
            int(label)
            for matrix in matrices
            for label in matrix.outputs
            if isinstance(label, (int, np.integer))
        }
    )
    output_labels = np.array(["lower_overflow", *all_outputs, "upper_overflow"], dtype=object)
    averaged = np.zeros((len(alphabet), len(output_labels)))
    for weight, matrix in zip(weights, matrices, strict=True):
        lookup = {label: i for i, label in enumerate(matrix.outputs)}
        for col, label in enumerate(output_labels):
            if label in lookup:
                averaged[:, col] += weight * matrix.probabilities[:, lookup[label]]
    return ChannelMatrix(
        np.asarray(alphabet, dtype=float),
        output_labels,
        averaged,
        {"construction": "phase_gauss_legendre", "quadrature_points": points},
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
        estimates[i, labels["lower_overflow"]] = np.mean(indexes < lo)
        estimates[i, labels["upper_overflow"]] = np.mean(indexes > hi)
        for index in range(lo, hi + 1):
            estimates[i, labels[index]] = np.mean(indexes == index)
    return estimates
