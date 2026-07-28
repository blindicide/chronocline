"""Uniform timing quantizers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class UniformQuantizer:
    """Uniform floor or nearest quantizer with phase normalized modulo its step."""

    step: float
    phase: float = 0.0
    mode: str = "floor"

    def __post_init__(self) -> None:
        if self.step <= 0 or self.mode not in {"floor", "nearest"}:
            raise ValueError("step must be positive and mode must be floor or nearest")
        object.__setattr__(self, "phase", self.phase % self.step)

    def bin_index(self, value: ArrayLike) -> NDArray[np.int64]:
        """Return integer quantization bin indices."""
        scaled = (np.asarray(value, dtype=float) - self.phase) / self.step
        return (np.floor(scaled) if self.mode == "floor" else np.rint(scaled)).astype(np.int64)

    def boundaries(self, index: int) -> tuple[float, float]:
        """Return the half-open bin interval for a floor quantizer."""
        if self.mode != "floor":
            center = self.phase + index * self.step
            return center - self.step / 2, center + self.step / 2
        lower = self.phase + index * self.step
        return lower, lower + self.step

    def quantize(self, value: ArrayLike) -> NDArray[np.float64]:
        """Quantize scalar or array values vectorially."""
        return np.asarray(self.phase + self.step * self.bin_index(value), dtype=np.float64)

    def scalar(self, value: float) -> float:
        """Quantize one value."""
        return float(self.quantize(value))

    def metadata(self) -> dict[str, float | str]:
        """Return a machine-readable exact quantizer definition."""
        return {"type": "uniform", "step": self.step, "phase": self.phase, "mode": self.mode}
