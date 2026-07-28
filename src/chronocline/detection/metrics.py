"""Detector empirical metrics."""

from __future__ import annotations

import numpy as np


def roc(scores0: np.ndarray, scores1: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return FPR, TPR, and trapezoidal AUC over observed thresholds."""
    thresholds = np.r_[np.inf, np.unique(np.r_[scores0, scores1])[::-1], -np.inf]
    fpr = np.array([(scores0 >= t).mean() for t in thresholds])
    tpr = np.array([(scores1 >= t).mean() for t in thresholds])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))
