"""Equivalent mutual-information implementations for cross-validation."""

from __future__ import annotations

import numpy as np

from .entropy import entropy, validate_distribution


def mutual_information(probabilities: np.ndarray, channel: np.ndarray) -> float:
    """Compute I(X;Y) directly in bits for a row-stochastic channel."""
    p = validate_distribution(probabilities)
    w = np.asarray(channel, dtype=float)
    if w.shape[0] != len(p):
        raise ValueError("input probability and channel dimensions differ")
    q = p @ w
    mask = (w > 0) & (q[None, :] > 0)
    terms = np.zeros_like(w)
    ratio = np.ones_like(w)
    ratio[mask] = w[mask] / np.broadcast_to(q, w.shape)[mask]
    weighted = p[:, None] * w * np.log2(ratio)
    terms[mask] = weighted[mask]
    return float(np.sum(terms))


def mutual_information_entropy(probabilities: np.ndarray, channel: np.ndarray) -> float:
    """Compute I(X;Y)=H(Y)-H(Y|X) in bits."""
    p = validate_distribution(probabilities)
    w = np.asarray(channel, dtype=float)
    return entropy(p @ w) - float(sum(p_i * entropy(row) for p_i, row in zip(p, w, strict=True)))
