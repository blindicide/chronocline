"""Controlled Monte Carlo cover-perturbation channel estimates."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..quantization import UniformQuantizer
from .matrix import ChannelMatrix


def cover_perturbation_matrix(
    baseline_samples: np.ndarray,
    perturbations: np.ndarray,
    jitter_sampler: Callable[[int, np.random.Generator], np.ndarray],
    quantizer: UniformQuantizer,
    rng: np.random.Generator,
    samples: int = 100_000,
) -> ChannelMatrix:
    """Estimate a cover-perturbation channel and label it explicitly as Monte Carlo."""
    base = rng.choice(np.asarray(baseline_samples, dtype=float), samples)
    raw = [quantizer.bin_index(base + u + jitter_sampler(samples, rng)) for u in perturbations]
    support = np.unique(np.concatenate(raw))
    matrix = np.array([np.array([(row == label).mean() for label in support]) for row in raw])
    return ChannelMatrix(
        np.asarray(perturbations, dtype=float),
        support,
        matrix,
        {"construction": "monte_carlo", "samples": samples},
    )
