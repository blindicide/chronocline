"""Bootstrap confidence intervals."""

import numpy as np


def bootstrap_mean(
    values: np.ndarray, repetitions: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Return a percentile 95% bootstrap interval for a sample mean."""
    sample = np.asarray(values, float)
    means = [rng.choice(sample, len(sample)).mean() for _ in range(repetitions)]
    return tuple(np.quantile(means, [0.025, 0.975]))  # type: ignore[return-value]
