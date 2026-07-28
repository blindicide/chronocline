"""Experiment-specific plotting from stored schema-2 result data."""

import json
from pathlib import Path

import pandas as pd

from .alphabet import plot_alphabet
from .batching import plot_batching
from .capacity import plot_capacity, plot_capacity_surface
from .detectability import plot_frontier
from .detection import plot_detection
from .jitter import plot_jitter
from .memoryless import plot_memoryless_summary
from .phase import plot_phase


def plot_result_directory(path: Path, locale: str = "en") -> Path:
    """Dispatch plots by manifest experiment kind, never by row index."""
    if (path / "LATEST").exists():
        path = path / (path / "LATEST").read_text().strip()
    manifest = json.loads((path / "manifest.json").read_text())
    results = pd.read_csv(path / "results.csv")
    figures = path / "figures"
    kind = manifest["experiment_kind"]
    if kind == "capacity_surface":
        return plot_capacity_surface(results, figures, locale)
    if kind == "phase_sensitivity":
        return plot_phase(results, figures)
    if kind == "detectability_frontier":
        return plot_frontier(results, figures)
    if kind == "finite_sample_detection":
        return plot_detection(results, figures)
    if kind == "alphabet_optimization":
        return plot_alphabet(results, figures)
    if kind == "batching_comparison":
        return plot_batching(results, figures)
    if kind == "jitter_comparison":
        return plot_jitter(results, figures)
    if kind in {"smoke", "memoryless_baseline"}:
        return plot_memoryless_summary(results, figures)
    return plot_capacity(results, figures, locale)


__all__ = ["plot_capacity", "plot_result_directory"]
