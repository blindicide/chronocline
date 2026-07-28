"""Atomic tidy-table result storage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_results(directory: Path, rows: list[dict[str, object]]) -> Path:
    """Write deterministic CSV and optional Parquet result tables atomically."""
    frame = pd.DataFrame(rows).sort_index(axis=1)
    temporary = directory / "results.csv.tmp"
    frame.to_csv(temporary, index=False)
    target = directory / "results.csv"
    temporary.replace(target)
    try:
        frame.to_parquet(directory / "results.parquet", index=False)
    except ImportError:
        pass
    return target
