"""Feasible ordered timing-alphabet optimization primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution


@dataclass(frozen=True)
class RestartCandidate:
    """One deterministic global-search attempt retained for auditability."""

    seed: int
    alphabet: np.ndarray
    objective: float
    accepted: bool


@dataclass(frozen=True)
class AlphabetResult:
    """A feasible exact-grid or best-found numerical alphabet solution."""

    alphabet: np.ndarray
    objective: float
    label: str
    restarts: tuple[RestartCandidate, ...] = field(default_factory=tuple)


def _valid(alphabet: np.ndarray, minimum: float, maximum: float, min_spacing: float) -> bool:
    return bool(
        len(alphabet) >= 2
        and alphabet[0] >= minimum
        and alphabet[-1] <= maximum
        and np.all(np.diff(alphabet) >= min_spacing)
    )


def binary_grid_optimize(
    minimum: float,
    maximum: float,
    points: int,
    objective: Callable[[np.ndarray], float],
    *,
    min_spacing: float = 0.0,
    anchor_first_symbol: bool = False,
) -> AlphabetResult:
    """Find a binary optimum exactly on the declared grid and constraints."""
    if points < 2 or maximum - minimum < min_spacing:
        raise ValueError("invalid binary alphabet grid")
    grid = np.linspace(minimum, maximum, points)
    first_values = np.array([minimum]) if anchor_first_symbol else grid
    candidates = [
        np.array([first, second])
        for first in first_values
        for second in grid
        if second - first >= min_spacing
    ]
    if not candidates:
        raise ValueError("binary grid contains no feasible alphabet")
    value, alphabet = max(
        ((float(objective(item)), item) for item in candidates), key=lambda item: item[0]
    )
    label = "binary_dense_grid_global_on_grid" if anchor_first_symbol else "exact_grid_optimum"
    return AlphabetResult(alphabet, value, label)


def best_found_alphabet(
    symbols: int,
    minimum: float,
    maximum: float,
    min_spacing: float,
    objective: Callable[[np.ndarray], float],
    seed: int = 0,
    *,
    anchor_first_symbol: bool = True,
    global_restarts: int = 1,
) -> AlphabetResult:
    """Best-found ordered alphabet with slack represented by absolute positions.

    Unlike a gap-normalisation parameterisation, this search does not force the
    final symbol to ``maximum``.  Invalid candidates receive an infinite
    minimisation penalty, so feasibility changes selection rather than being a
    cosmetic post-processing check.
    """
    if symbols < 2 or global_restarts < 1 or maximum - minimum < min_spacing * (symbols - 1):
        raise ValueError("invalid alphabet bounds, spacing, or restart count")
    dimensions = symbols - 1 if anchor_first_symbol else symbols

    def decode(values: np.ndarray) -> np.ndarray:
        return np.r_[minimum, values] if anchor_first_symbol else values

    def loss(values: np.ndarray) -> float:
        alphabet = decode(values)
        if not _valid(alphabet, minimum, maximum, min_spacing):
            return 1e100
        value = float(objective(alphabet))
        return -value if np.isfinite(value) else 1e100

    seeds = np.random.SeedSequence(seed).spawn(global_restarts)
    candidates: list[RestartCandidate] = []
    for child in seeds:
        restart_seed = int(child.generate_state(1)[0])
        result = differential_evolution(
            loss,
            [(minimum, maximum)] * dimensions,
            seed=restart_seed,
            polish=True,
        )
        alphabet = decode(result.x)
        accepted = _valid(alphabet, minimum, maximum, min_spacing) and result.fun < 1e99
        candidates.append(
            RestartCandidate(
                restart_seed,
                alphabet,
                -float(result.fun) if accepted else -np.inf,
                accepted,
            )
        )
    accepted_candidates = [item for item in candidates if item.accepted]
    if not accepted_candidates:
        raise RuntimeError("no feasible alphabet was accepted by global optimization")
    best = max(accepted_candidates, key=lambda item: item.objective)
    return AlphabetResult(
        best.alphabet,
        best.objective,
        "ternary_differential_evolution_best_found",
        tuple(candidates),
    )
