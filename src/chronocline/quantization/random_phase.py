"""Deterministic quadrature helpers for random quantizer phase."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss


def phase_nodes(step: float, points: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre phase nodes and normalized weights on [0, step]."""
    if step <= 0 or points < 2:
        raise ValueError("step must be positive and at least two quadrature points are required")
    nodes, weights = leggauss(points)
    return (nodes + 1) * step / 2, weights / 2
