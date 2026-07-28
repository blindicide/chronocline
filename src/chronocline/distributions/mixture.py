"""Finite Gaussian mixture jitter distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


@dataclass(frozen=True)
class GaussianMixture:
    """Gaussian mixture with explicitly normalized component weights."""

    weights: NDArray[np.float64]
    means: NDArray[np.float64]
    scales: NDArray[np.float64]

    def __init__(
        self, weights: ArrayLike, means: ArrayLike, scales: ArrayLike, *, normalize: bool = False
    ) -> None:
        w, m, s = (np.asarray(values, dtype=float) for values in (weights, means, scales))
        if len(w) == 0 or len(w) != len(m) or len(w) != len(s) or np.any(w < 0) or np.any(s <= 0):
            raise ValueError(
                "mixture arrays must have equal non-empty lengths, non-negative weights, "
                "and positive scales"
            )
        total = w.sum()
        if not np.isclose(total, 1.0):
            if not normalize or total <= 0:
                raise ValueError("mixture weights must sum to one unless normalize=True")
            w = w / total
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "means", m)
        object.__setattr__(self, "scales", s)

    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        a = np.asarray(x, dtype=float)[..., None]
        return np.sum(self.weights * stats.norm.pdf(a, self.means, self.scales), axis=-1)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        a = np.asarray(x, dtype=float)[..., None]
        return np.sum(self.weights * stats.norm.cdf(a, self.means, self.scales), axis=-1)

    def ppf(self, q: ArrayLike) -> NDArray[np.float64]:
        from scipy.optimize import brentq

        q_arr = np.asarray(q, dtype=float)
        if np.any((q_arr < 0) | (q_arr > 1)):
            raise ValueError("quantiles must be in [0, 1]")
        spread = max(float(np.max(self.scales)), 1.0)
        lo = float(np.min(self.means) - 12 * spread)
        hi = float(np.max(self.means) + 12 * spread)
        values = []
        for target in q_arr.ravel():
            if target == 0:
                values.append(-np.inf)
            elif target == 1:
                values.append(np.inf)
            else:
                values.append(
                    brentq(lambda z, value=target: float(self.cdf(z) - value), lo, hi)
                )
        return np.asarray(values).reshape(q_arr.shape)

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        indexes = rng.choice(len(self.weights), size=size, p=self.weights)
        return rng.normal(self.means[indexes], self.scales[indexes])

    def support(self) -> tuple[float, float]:
        return -np.inf, np.inf

    def mean(self) -> float:
        return float(self.weights @ self.means)

    def variance(self) -> float:
        mean = self.mean()
        return float(self.weights @ (self.scales**2 + (self.means - mean) ** 2))
