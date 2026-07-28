"""Stateful correlated jitter processes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AR1Jitter:
    """AR(1) process with configured stationary mean and variance."""

    rho: float
    mean_value: float = 0.0
    variance_value: float = 1.0

    def __post_init__(self) -> None:
        if abs(self.rho) >= 1 or self.variance_value <= 0:
            raise ValueError("AR(1) requires |rho| < 1 and positive variance")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generate n stationary samples; this process is not memoryless."""
        if n < 1:
            raise ValueError("n must be positive")
        out = np.empty(n)
        out[0] = rng.normal(self.mean_value, np.sqrt(self.variance_value))
        innovation_scale = np.sqrt(self.variance_value * (1 - self.rho**2))
        for i in range(1, n):
            out[i] = (
                self.mean_value
                + self.rho * (out[i - 1] - self.mean_value)
                + rng.normal(0, innovation_scale)
            )
        return out
