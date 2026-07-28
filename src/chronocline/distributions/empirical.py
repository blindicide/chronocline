"""Measured jitter distribution without extrapolation outside its sample support."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class EmpiricalJitter:
    """Empirical CDF and bootstrap sampler from finite measured samples."""

    samples: NDArray[np.float64]
    source_hash: str | None = None

    def __init__(self, samples: ArrayLike, source_hash: str | None = None) -> None:
        values = np.sort(np.asarray(samples, dtype=float))
        if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError("samples must be a non-empty finite one-dimensional array")
        object.__setattr__(self, "samples", values)
        object.__setattr__(self, "source_hash", source_hash)

    @classmethod
    def from_file(cls, path: str | Path, column: str | None = None) -> EmpiricalJitter:
        """Load samples from CSV or Parquet and record a SHA-256 content hash."""
        path = Path(path)
        raw = path.read_bytes()
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        series = frame[column] if column else frame.iloc[:, 0]
        return cls(series.to_numpy(), sha256(raw).hexdigest())

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(x, dtype=float)
        return np.searchsorted(self.samples, values, side="right") / len(self.samples)

    def ppf(self, q: ArrayLike) -> NDArray[np.float64]:
        qs = np.asarray(q, dtype=float)
        if np.any((qs < 0) | (qs > 1)):
            raise ValueError("quantiles must be in [0, 1]")
        return np.quantile(self.samples, qs)

    def sample(self, size: int | tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.choice(self.samples, size=size)

    def support(self) -> tuple[float, float]:
        return float(self.samples[0]), float(self.samples[-1])

    def mean(self) -> float:
        return float(np.mean(self.samples))

    def variance(self) -> float:
        return float(np.var(self.samples))
