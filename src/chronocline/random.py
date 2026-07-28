"""Deterministic random-stream helpers."""

from __future__ import annotations

import numpy as np


def generator(seed: int, stream: int = 0) -> np.random.Generator:
    """Return a reproducible child generator for a non-negative seed and stream."""
    if seed < 0 or stream < 0:
        raise ValueError("seed and stream must be non-negative")
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(stream,)))
