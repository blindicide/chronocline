"""Simplex grids used to independently validate small constrained problems."""

from __future__ import annotations

import numpy as np


def simplex_grid(dimensions: int, resolution: int) -> np.ndarray:
    """Return deterministic simplex grid points for two or three symbols."""
    if dimensions not in {2, 3} or resolution < 1:
        raise ValueError("grid supports two or three dimensions with positive resolution")
    values = np.linspace(0, 1, resolution + 1)
    if dimensions == 2:
        return np.c_[values, 1 - values]
    return np.array([[a, b, 1 - a - b] for a in values for b in values if a + b <= 1])
