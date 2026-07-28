"""Safe plot writing and source-table helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def require_variation(frame: pd.DataFrame, column: str) -> None:
    """Require a real scientific axis rather than an implicit row index."""
    if column not in frame or frame[column].nunique() < 2:
        raise ValueError(f"plot requires multiple values of {column}")


def save_figure(figure: plt.Figure, directory: Path, name: str, source: pd.DataFrame) -> Path:
    """Write vector/raster outputs and their machine-readable source table."""
    directory.mkdir(parents=True, exist_ok=True)
    table_directory = directory.parent / "tables"
    table_directory.mkdir(exist_ok=True)
    source.to_csv(table_directory / f"figure_{name}.csv", index=False)
    for suffix in ("png", "pdf"):
        figure.savefig(directory / f"{name}.{suffix}", dpi=300 if suffix == "png" else None)
    plt.close(figure)
    return directory / f"{name}.png"
