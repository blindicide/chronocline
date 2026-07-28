"""Validated row-stochastic discrete channel matrices."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import PROBABILITY_TOLERANCE
from ..exceptions import ValidationError


@dataclass(frozen=True)
class ChannelMatrix:
    """Immutable finite channel with rows indexed by inputs and shared output labels."""

    inputs: np.ndarray
    outputs: np.ndarray
    probabilities: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        p = np.asarray(self.probabilities, dtype=float)
        if p.ndim != 2 or p.shape != (len(self.inputs), len(self.outputs)):
            raise ValidationError("matrix shape must match input and output supports")
        if np.any(p < -PROBABILITY_TOLERANCE):
            raise ValidationError("channel probabilities must be non-negative")
        residual = np.max(np.abs(p.sum(axis=1) - 1))
        if residual > PROBABILITY_TOLERANCE:
            raise ValidationError(
                f"channel rows must sum to one; maximum residual is {residual:.3e}"
            )
        object.__setattr__(self, "inputs", np.asarray(self.inputs, dtype=float))
        object.__setattr__(self, "outputs", np.asarray(self.outputs))
        object.__setattr__(self, "probabilities", np.maximum(p, 0.0))

    @property
    def row_sum_error(self) -> float:
        """Return the maximum stochasticity residual."""
        return float(np.max(np.abs(self.probabilities.sum(axis=1) - 1)))
