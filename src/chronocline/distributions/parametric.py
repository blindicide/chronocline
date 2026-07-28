"""Parametric independent jitter distributions backed by SciPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


@dataclass(frozen=True)
class ScipyJitter:
    """Thin immutable adapter around a SciPy continuous distribution."""

    law: Any
    name: str

    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(self.law.pdf(x), dtype=float)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(self.law.cdf(x), dtype=float)

    def ppf(self, q: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(self.law.ppf(q), dtype=float)

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        return np.asarray(self.law.rvs(size=size, random_state=rng), dtype=float)

    def support(self) -> tuple[float, float]:
        lo, hi = self.law.support()
        return float(lo), float(hi)

    def mean(self) -> float:
        return float(self.law.mean())

    def variance(self) -> float:
        return float(self.law.var())


def gaussian(mean: float = 0.0, scale: float = 1.0) -> ScipyJitter:
    """Create Gaussian jitter with standard deviation ``scale``."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    return ScipyJitter(stats.norm(loc=mean, scale=scale), "gaussian")


def laplace(mean: float = 0.0, scale: float = 1.0) -> ScipyJitter:
    """Create Laplace jitter with positive scale."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    return ScipyJitter(stats.laplace(loc=mean, scale=scale), "laplace")


def uniform(lower: float, upper: float) -> ScipyJitter:
    """Create uniform jitter on the closed numerical interval [lower, upper]."""
    if upper <= lower:
        raise ValueError("upper must exceed lower")
    return ScipyJitter(stats.uniform(loc=lower, scale=upper - lower), "uniform")


def student_t(degrees_of_freedom: float, mean: float = 0.0, scale: float = 1.0) -> ScipyJitter:
    """Create location-scale Student t jitter."""
    if degrees_of_freedom <= 0 or scale <= 0:
        raise ValueError("degrees_of_freedom and scale must be positive")
    return ScipyJitter(stats.t(df=degrees_of_freedom, loc=mean, scale=scale), "student_t")
