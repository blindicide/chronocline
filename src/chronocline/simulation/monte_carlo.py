"""Empirical information estimates for stateful traces."""

from __future__ import annotations

import numpy as np


def empirical_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Plug-in I(X;Y) for discrete samples; not channel capacity with memory."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) != len(y):
        raise ValueError("sample vectors must have equal length")
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((xi.max() + 1, yi.max() + 1))
    np.add.at(joint, (xi, yi), 1)
    joint /= len(x)
    px, py = joint.sum(1), joint.sum(0)
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / (px[:, None] * py)[mask])))


def block_mutual_information(x: np.ndarray, y: np.ndarray, block: int) -> dict[str, float]:
    """Return plug-in block MI and normalized estimate with Miller-Madow warning metadata."""
    if block < 1:
        raise ValueError("block must be positive")
    n = len(x) // block
    xb = np.asarray(x)[: n * block].reshape(n, block)
    yb = np.asarray(y)[: n * block].reshape(n, block)
    _, x_codes = np.unique(xb, axis=0, return_inverse=True)
    _, y_codes = np.unique(yb, axis=0, return_inverse=True)
    value = empirical_mutual_information(x_codes, y_codes)
    return {
        "block_length": block,
        "block_mutual_information_estimate": value,
        "normalized_block_estimate": value / block,
        "warning_small_sample": float(n < 10 * len(np.unique(xb, axis=0))),
    }
