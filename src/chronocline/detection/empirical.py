"""Finite-sample independent detector experiments."""

from __future__ import annotations

import numpy as np

from .likelihood_ratio import log_likelihood_ratio
from .metrics import roc


def detection_trial(
    active: np.ndarray, baseline: np.ndarray, n: int, trials: int, rng: np.random.Generator
) -> dict[str, object]:
    """Simulate likelihood-ratio scores under active and baseline hypotheses."""
    p1, p0 = np.asarray(active, float), np.asarray(baseline, float)
    s0 = np.array(
        [log_likelihood_ratio(rng.choice(len(p0), n, p=p0), p1, p0) for _ in range(trials)]
    )
    s1 = np.array(
        [log_likelihood_ratio(rng.choice(len(p1), n, p=p1), p1, p0) for _ in range(trials)]
    )
    fpr, tpr, auc = roc(s0, s1)
    return {
        "n": n,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "minimum_total_error": float(np.min((fpr + (1 - tpr)) / 2)),
    }
