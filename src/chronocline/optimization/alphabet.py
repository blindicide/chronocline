"""Best-found ordered timing alphabet search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution


@dataclass(frozen=True)
class AlphabetResult:
    """A best-found (or exact binary-grid) alphabet optimization result."""

    alphabet: np.ndarray
    objective: float
    label: str


def binary_grid_optimize(
    minimum: float, maximum: float, points: int, objective: Callable[[np.ndarray], float]
) -> AlphabetResult:
    """Find an exact-on-grid binary alphabet optimum while enforcing strict order."""
    grid = np.linspace(minimum, maximum, points)
    candidates = [
        (float(objective(np.array([a, b]))), np.array([a, b])) for a in grid for b in grid if b > a
    ]
    value, alphabet = max(candidates, key=lambda item: item[0])
    return AlphabetResult(alphabet, value, "exact_grid_optimum")


def best_found_alphabet(
    symbols: int,
    minimum: float,
    maximum: float,
    min_spacing: float,
    objective: Callable[[np.ndarray], float],
    seed: int = 0,
) -> AlphabetResult:
    """Use differential evolution over gaps and explicitly label output best-found."""
    if symbols < 2 or maximum - minimum < min_spacing * (symbols - 1):
        raise ValueError("invalid alphabet bounds or spacing")

    def decode(gaps: np.ndarray) -> np.ndarray:
        extras = (maximum - minimum - min_spacing * (symbols - 1)) * gaps / max(gaps.sum(), 1e-15)
        return minimum + np.r_[0.0, np.cumsum(min_spacing + extras)]

    result = differential_evolution(
        lambda x: -float(objective(decode(x))), [(1e-8, 1)] * (symbols - 1), seed=seed, polish=True
    )
    return AlphabetResult(decode(result.x), -float(result.fun), "best_found_numerical_solution")
