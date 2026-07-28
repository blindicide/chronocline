"""Typed experiment dispatch, clean provenance, and schema-2 storage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from ..config import ExperimentKind, RunConfig
from ..results.manifest import (
    configuration_hash,
    create_manifest,
    finalize_manifest,
    git_state,
    run_identifier,
    write_environment,
)
from ..results.storage import write_results
from ..results.validation import semantic_errors
from .alphabet import AlphabetRunner
from .base import ExperimentContext, ExperimentPlan, ExperimentRunner
from .batching import BatchingRunner
from .constrained import ConstrainedRunner
from .detection import DetectionRunner
from .memoryless import MemorylessRunner
from .phase import PhaseRunner
from .sweep import resolve_sweep

RUNNERS: dict[ExperimentKind, ExperimentRunner] = {
    ExperimentKind.SMOKE: MemorylessRunner(),
    ExperimentKind.MEMORYLESS_BASELINE: MemorylessRunner(),
    ExperimentKind.CAPACITY_CURVE: MemorylessRunner(),
    ExperimentKind.CAPACITY_SURFACE: MemorylessRunner(),
    ExperimentKind.JITTER_COMPARISON: MemorylessRunner(),
    ExperimentKind.PHASE_SENSITIVITY: PhaseRunner(),
    ExperimentKind.DETECTABILITY_FRONTIER: ConstrainedRunner(),
    ExperimentKind.FINITE_SAMPLE_DETECTION: DetectionRunner(),
    ExperimentKind.BATCHING_COMPARISON: BatchingRunner(),
    ExperimentKind.ALPHABET_OPTIMIZATION: AlphabetRunner(),
}

ALLOWED_SWEEPS = {
    ExperimentKind.SMOKE: {"quantizer.step"},
    ExperimentKind.MEMORYLESS_BASELINE: {"quantizer.step"},
    ExperimentKind.CAPACITY_CURVE: {"quantizer.step"},
    ExperimentKind.CAPACITY_SURFACE: {"quantizer.step", "channel.alphabet.values"},
    ExperimentKind.JITTER_COMPARISON: {"quantizer.step", "jitter.distribution", "jitter.scale"},
    ExperimentKind.PHASE_SENSITIVITY: {"quantizer.phase"},
    ExperimentKind.DETECTABILITY_FRONTIER: {"constraints.max_kl_bits"},
    ExperimentKind.FINITE_SAMPLE_DETECTION: set(),
    ExperimentKind.BATCHING_COMPARISON: set(),
    ExperimentKind.ALPHABET_OPTIMIZATION: set(),
}


def plan(config: RunConfig) -> ExperimentPlan:
    """Resolve runner planning without generating output."""
    root = Path(config.experiment.output_directory) / config.experiment.name
    return RUNNERS[config.experiment.kind].plan(config, root)


def run(
    config: RunConfig, *, dry_run: bool = False, allow_dirty: bool = False
) -> Path | dict[str, object]:
    """Execute the runner matching ``experiment.kind`` and validate its output."""
    source_commit, source_dirty = git_state()
    if config.experiment.require_clean_git and source_dirty and not allow_dirty:
        raise RuntimeError(
            "publication experiment requires a clean Git source tree; "
            "use --allow-dirty only for development"
        )
    resolved = resolve_sweep(config, ALLOWED_SWEEPS[config.experiment.kind])
    config_hash = configuration_hash(config.model_dump(mode="json"))
    run_id = run_identifier(config_hash, source_commit, config.experiment.kind)
    directory = Path(config.experiment.output_directory) / config.experiment.name / run_id
    runner = RUNNERS[config.experiment.kind]
    experiment_plan = runner.plan(config, directory)
    if dry_run:
        return {
            "kind": config.experiment.kind,
            "jobs": len(resolved),
            "expected_metrics": sorted(experiment_plan.expected_metrics),
            "output_directory": str(directory),
            "workers": config.experiment.workers,
        }
    if directory.exists() and not config.experiment.overwrite:
        errors = semantic_errors(directory, strict=True)
        if not errors:
            return directory
    manifest = create_manifest(
        run_id=run_id,
        config_hash=config_hash,
        experiment_name=config.experiment.name,
        experiment_kind=config.experiment.kind,
        runner_name=type(runner).__name__,
        source_commit=source_commit,
        source_dirty=source_dirty,
        allow_dirty_override=allow_dirty or not config.experiment.require_clean_git,
        workers=config.experiment.workers,
        expected_jobs=experiment_plan.jobs,
        expected_metrics=sorted(experiment_plan.expected_metrics),
        locale=config.experiment.locale,
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.original.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )
    (directory / "config.resolved.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True)
    )
    write_environment(directory, manifest)
    context = ExperimentContext(
        config,
        directory,
        np.random.SeedSequence(config.experiment.seed),
        source_commit,
        source_dirty,
    )
    output = runner.execute(context, [(index, job) for index, job, _ in resolved])
    write_results(directory, output.rows)
    work_units = _work_units(output.rows, config.experiment.kind)
    tables = directory / "tables"
    tables.mkdir(exist_ok=True)
    import pandas as pd

    pd.DataFrame(work_units).to_csv(tables / "work_units.csv", index=False)
    (directory / "diagnostics.json").write_text(
        json.dumps(output.diagnostics, indent=2, default=str)
    )
    errors = semantic_errors(directory, manifest=manifest, strict=False)
    finalize_manifest(directory, manifest, len(work_units), errors)
    if errors:
        raise RuntimeError("semantic validation failed: " + "; ".join(errors))
    (directory.parent / "LATEST").write_text(run_id, encoding="utf-8")
    return directory


def _work_units(rows: list[dict[str, object]], kind: ExperimentKind) -> list[dict[str, object]]:
    """Derive one auditable scientific work unit per emitted schema-2 job identifier."""
    units: dict[str, dict[str, object]] = {}
    parameter_keys = (
        "sample_size",
        "hypothesis_pair",
        "replication",
        "batching_mode",
        "batching_window",
        "restart_index",
        "sweep_index",
    )
    for item in rows:
        if kind is ExperimentKind.FINITE_SAMPLE_DETECTION:
            identifier = ":".join(
                str(item.get(key, "")) for key in ("sample_size", "hypothesis_pair", "replication")
            )
        elif kind is ExperimentKind.BATCHING_COMPARISON:
            identifier = ":".join(
                str(item.get(key, ""))
                for key in ("batching_mode", "batching_window", "replication")
            )
        elif kind is ExperimentKind.ALPHABET_OPTIMIZATION:
            identifier = str(item.get("restart_index", 0))
        else:
            identifier = str(item["job_id"])
        if identifier not in units:
            units[identifier] = {
                "work_unit_id": identifier,
                "experiment_kind": str(kind),
                "status": str(item.get("status", "complete")),
                "error": "",
                **{key: item.get(key) for key in parameter_keys if key in item},
            }
    return [units[key] for key in sorted(units)]
