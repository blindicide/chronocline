"""Pareto filtering for capacity-detectability frontiers."""

import numpy as np


def nondominated(points: np.ndarray) -> np.ndarray:
    """Keep points maximizing first coordinate and minimizing second coordinate."""
    values = np.asarray(points, float)
    keep = np.ones(len(values), dtype=bool)
    for i, point in enumerate(values):
        keep[i] = not any(
            (other[0] >= point[0] and other[1] <= point[1]) and np.any(other != point)
            for other in values
        )
    return values[keep]
