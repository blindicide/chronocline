"""Typed experiment contracts shared by specialised runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from ..config import ExperimentKind, RunConfig


@dataclass(frozen=True)
class ExperimentPlan:
    """Expected work and artifacts for one specialised experiment run."""

    kind: ExperimentKind
    jobs: int
    expected_metrics: frozenset[str]
    expected_artifacts: frozenset[str]
    output_directory: Path


@dataclass(frozen=True)
class ExperimentContext:
    """Execution context with fixed provenance and root RNG stream."""

    config: RunConfig
    directory: Path
    root_seed_sequence: np.random.SeedSequence
    source_commit: str | None
    source_dirty: bool


@dataclass
class ExperimentOutput:
    """Rows, diagnostics, and relative artifacts produced by a runner."""

    rows: list[dict[str, object]] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)


class ExperimentRunner(Protocol):
    """Protocol required by the experiment runner registry."""

    def plan(self, config: RunConfig, output_directory: Path) -> ExperimentPlan: ...

    def execute(
        self, context: ExperimentContext, jobs: list[tuple[int, RunConfig]]
    ) -> ExperimentOutput: ...
