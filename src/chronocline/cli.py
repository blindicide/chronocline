"""Typer CLI for configuration validation, calculation, experiments, and result checks."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import ExperimentKind, load_config
from .experiments import run
from .plotting import plot_result_directory
from .results import validate_result_directory
from .results.manifest import finalize_manifest
from .results.validation import semantic_errors

app = typer.Typer(
    no_args_is_help=True, help="Project Chronocline scientific timing-channel framework."
)


@app.command("validate-config")
def validate_config(path: Path) -> None:
    """Validate a strict YAML configuration."""
    config = load_config(path)
    typer.echo(f"valid: {config.experiment.name}")


@app.command()
def experiment(path: Path, dry_run: bool = False, allow_dirty: bool = False) -> None:
    """Run a reproducible memoryless experiment or show its resolved work."""
    result = run(load_config(path), dry_run=dry_run, allow_dirty=allow_dirty)
    typer.echo(str(result))


@app.command()
def matrix(path: Path) -> None:
    """Run a matrix-producing experiment."""
    config = load_config(path)
    if config.experiment.kind not in {
        ExperimentKind.SMOKE,
        ExperimentKind.MEMORYLESS_BASELINE,
        ExperimentKind.CAPACITY_CURVE,
        ExperimentKind.CAPACITY_SURFACE,
        ExperimentKind.PHASE_SENSITIVITY,
        ExperimentKind.JITTER_COMPARISON,
    }:
        raise typer.BadParameter("matrix requires a memoryless-compatible experiment kind")
    typer.echo(str(run(config)))


@app.command()
def capacity(path: Path) -> None:
    """Run a capacity-producing experiment."""
    config = load_config(path)
    if config.experiment.kind in {
        ExperimentKind.FINITE_SAMPLE_DETECTION,
        ExperimentKind.BATCHING_COMPARISON,
    }:
        raise typer.BadParameter(
            "capacity is incompatible with detector-only or stateful simulation kinds"
        )
    typer.echo(str(run(config)))


@app.command("validate-results")
def validate_results(path: Path) -> None:
    """Validate stored checksums and required result files."""
    errors = validate_result_directory(path, strict=True)
    if errors:
        typer.echo("\n".join(errors))
        raise typer.Exit(1)
    typer.echo("valid")


@app.command()
def plot(path: Path, locale: str = "en") -> None:
    """Regenerate plots from stored CSV data only."""
    directory = path / (path / "LATEST").read_text().strip() if (path / "LATEST").exists() else path
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise typer.BadParameter("plot requires a computation manifest")
    manifest = json.loads(manifest_path.read_text())
    prior_errors = list(manifest.get("semantic_validation_errors", []))
    if (
        manifest.get("completion_status") == "failed"
        or manifest.get("semantic_validation_status") == "failed"
        or prior_errors
    ):
        raise typer.BadParameter("refusing to plot a failed or semantically invalid computation")
    try:
        result = plot_result_directory(path, locale)
    except Exception as error:
        finalize_manifest(
            directory,
            manifest,
            int(manifest.get("completed_jobs", 0)),
            [f"plotting failed: {error}"],
        )
        raise
    is_schema2_manifest = manifest.get("result_schema_version") is not None
    errors = semantic_errors(directory, strict=False) if is_schema2_manifest else []
    finalize_manifest(directory, manifest, int(manifest.get("completed_jobs", 0)), errors)
    if errors:
        raise typer.Exit(1)
    typer.echo(result)


@app.command("run-suite")
def run_suite(directory: Path) -> None:
    """Run every YAML configuration in deterministic filename order."""
    for path in sorted(directory.glob("*.yaml")):
        typer.echo(f"running {path}")
        run(load_config(path))


@app.command()
def simulate(path: Path) -> None:
    """Run a configured simulation experiment."""
    config = load_config(path)
    if config.experiment.kind is not ExperimentKind.BATCHING_COMPARISON:
        raise typer.BadParameter("simulate requires batching_comparison")
    typer.echo(str(run(config)))


@app.command()
def detect(path: Path) -> None:
    """Run a configured detection experiment."""
    config = load_config(path)
    if config.experiment.kind is not ExperimentKind.FINITE_SAMPLE_DETECTION:
        raise typer.BadParameter("detect requires finite_sample_detection")
    typer.echo(str(run(config)))


@app.command("constrained-capacity")
def constrained_capacity_command(path: Path) -> None:
    """Run a configured constrained-capacity experiment."""
    config = load_config(path)
    if config.experiment.kind is not ExperimentKind.DETECTABILITY_FRONTIER:
        raise typer.BadParameter("constrained-capacity requires detectability_frontier")
    typer.echo(str(run(config)))
