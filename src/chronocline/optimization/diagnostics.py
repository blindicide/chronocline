"""Optimization diagnostics utilities."""

import numpy as np


def ordered(alphabet: np.ndarray, spacing: float = 0.0) -> bool:
    """Check strict alphabet ordering with a minimum gap."""
    return bool(np.all(np.diff(np.asarray(alphabet, float)) >= spacing))
