"""Integrity and schema-2 semantic result validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import METRIC_UNITS

REQUIRED = {
    "memoryless_baseline": {
        "mutual_information",
        "capacity_bits_per_symbol",
        "capacity_residual",
        "matrix_row_sum_error",
        "monte_carlo_max_absolute_error",
    },
    "smoke": {
        "mutual_information",
        "capacity_bits_per_symbol",
        "matrix_row_sum_error",
        "monte_carlo_max_absolute_error",
    },
    "capacity_surface": {"capacity_bits_per_symbol"},
    "phase_sensitivity": {"quantizer_phase", "capacity_bits_per_symbol"},
    "detectability_frontier": {
        "constrained_capacity_bits_per_symbol",
        "achieved_kl_bits",
        "mean_delay",
        "optimizer_converged",
        "optimizer_feasible",
    },
    "finite_sample_detection": {"auc", "minimum_equal_prior_error"},
    "alphabet_optimization": {"optimized_alphabet", "best_found_capacity", "optimization_label"},
    "batching_comparison": {
        "symbol_mutual_information",
        "zero_delay_probability",
        "batch_size_mean",
        "memoryless_approximation_error",
        "plugin_block_mutual_information",
    },
}


def _resolve(path: Path) -> Path:
    latest = path / "LATEST"
    return path / latest.read_text().strip() if latest.exists() else path


def semantic_errors(
    path: str | Path, manifest: dict[str, Any] | None = None, *, strict: bool = True
) -> list[str]:
    """Return semantic/integrity violations; an empty list means publication-valid."""
    directory = _resolve(Path(path))
    errors = []
    manifest_path = directory / "manifest.json"
    if manifest is None:
        if not manifest_path.exists():
            return ["missing manifest.json"]
        manifest = json.loads(manifest_path.read_text())
    results_path = directory / "results.csv"
    if not results_path.exists():
        return ["missing results.csv"]
    frame = pd.read_csv(results_path)
    required_columns = {
        "experiment_name",
        "experiment_kind",
        "job_id",
        "sweep_index",
        "metric_name",
        "metric_value",
        "units",
        "estimator",
        "status",
    }
    if not required_columns.issubset(frame.columns):
        errors.append("schema-2 result columns missing")
    if "metric_value" in frame:
        allowed_infinite = frame.metric_name.eq("theoretical_total_kl_bits")
        if not np.all(np.isfinite(frame.metric_value) | allowed_infinite):
            errors.append("non-finite result value")
    for metric, unit in zip(frame.get("metric_name", []), frame.get("units", []), strict=False):
        if METRIC_UNITS.get(metric) != unit:
            errors.append(f"wrong or unknown unit for {metric}")
    kind = str(manifest.get("experiment_kind", ""))
    required = REQUIRED.get(kind, {"capacity_bits_per_symbol"})
    if not required.issubset(set(frame.get("metric_name", []))):
        errors.append(f"missing required metrics for {kind}")
    if manifest.get("completed_jobs") not in {None, 0} and (
        manifest.get("completed_jobs") != manifest.get("expected_jobs")
    ):
        errors.append("incomplete jobs")
    if manifest.get("source_commit") is None:
        errors.append("missing source commit")
    if manifest.get("source_dirty") and not manifest.get("allow_dirty_override"):
        errors.append("dirty publication source")
    if kind == "capacity_surface" and (
        frame.get("quantizer_step", pd.Series()).nunique() < 2
        or frame.get("alphabet", pd.Series()).nunique() < 2
    ):
        errors.append("capacity surface lacks two varying axes")
    if (
        kind == "phase_sensitivity"
        and frame.loc[frame.metric_name == "quantizer_phase", "metric_value"].nunique() < 2
    ):
        errors.append("phase campaign lacks phase variation")
    if kind == "finite_sample_detection" and not list((directory / "tables").glob("roc_n_*.csv")):
        errors.append("detection campaign lacks ROC artifacts")
    if strict and manifest_path.exists():
        for name, expected in manifest.get("generated_files", {}).items():
            file = directory / name
            if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                errors.append(f"checksum mismatch: {name}")
    return sorted(set(errors))


def validate_result_directory(path: str | Path, strict: bool = True) -> list[str]:
    """Compatibility entry point for full semantic result validation."""
    return semantic_errors(path, strict=strict)
