"""Typer CLI for configuration validation, calculation, experiments, and result checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from .config import load_config
from .experiments import run
from .plotting import plot_capacity
from .results import validate_result_directory

app = typer.Typer(
    no_args_is_help=True, help="Project Chronocline scientific timing-channel framework."
)


@app.command("validate-config")
def validate_config(path: Path) -> None:
    """Validate a strict YAML configuration."""
    config = load_config(path)
    typer.echo(f"valid: {config.experiment.name}")


@app.command()
def experiment(path: Path, dry_run: bool = False) -> None:
    """Run a reproducible memoryless experiment or show its resolved work."""
    result = run(load_config(path), dry_run=dry_run)
    typer.echo(str(result))


@app.command()
def matrix(path: Path) -> None:
    """Run a matrix-producing experiment."""
    typer.echo(str(run(load_config(path))))


@app.command()
def capacity(path: Path) -> None:
    """Run a capacity-producing experiment."""
    typer.echo(str(run(load_config(path))))


@app.command("validate-results")
def validate_results(path: Path) -> None:
    """Validate stored checksums and required result files."""
    errors = validate_result_directory(path)
    if errors:
        typer.echo("\n".join(errors))
        raise typer.Exit(1)
    typer.echo("valid")


@app.command()
def plot(path: Path, locale: str = "en") -> None:
    """Regenerate plots from stored CSV data only."""
    plot_capacity(pd.read_csv(path / "results.csv"), path / "figures", locale)
    typer.echo(path / "figures")


@app.command("run-suite")
def run_suite(directory: Path) -> None:
    """Run every YAML configuration in deterministic filename order."""
    for path in sorted(directory.glob("*.yaml")):
        typer.echo(f"running {path}")
        run(load_config(path))


@app.command()
def simulate(path: Path) -> None:
    """Run a configured simulation experiment."""
    typer.echo(str(run(load_config(path))))


@app.command()
def detect(path: Path) -> None:
    """Run a configured detection experiment."""
    typer.echo(str(run(load_config(path))))


@app.command("constrained-capacity")
def constrained_capacity_command(path: Path) -> None:
    """Run a configured constrained-capacity experiment."""
    typer.echo(str(run(load_config(path))))
