"""Atomic tidy-table result storage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import ResultRow


def write_results(directory: Path, rows: list[dict[str, object]]) -> Path:
    """Write deterministic CSV and optional Parquet result tables atomically."""
    validated = [ResultRow.model_validate(item) for item in rows]
    for item in validated:
        item.validate_metric_unit()
    frame = pd.DataFrame([item.model_dump(mode="json") for item in validated]).sort_values(
        ["sweep_index", "metric_name"]
    )
    temporary = directory / "results.csv.tmp"
    frame.to_csv(temporary, index=False)
    target = directory / "results.csv"
    temporary.replace(target)
    try:
        frame.to_parquet(directory / "results.parquet", index=False)
    except ImportError:
        pass
    return target
