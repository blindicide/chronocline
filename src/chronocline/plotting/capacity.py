"""Capacity plot generation from completed result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .labels import label


def plot_capacity(results: pd.DataFrame, directory: str | Path, locale: str = "en") -> Path:
    """Write capacity curve in PNG, PDF, and SVG without recalculating data."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = results[results.metric_name == "capacity_bits_per_symbol"]
    x = data["quantizer.step"] if "quantizer.step" in data else data.replication
    figure, axis = plt.subplots(layout="constrained")
    axis.plot(x, data.metric_value, marker="o")
    axis.set(xlabel=label("step", locale), ylabel=label("capacity", locale))
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(directory / f"capacity.{suffix}", dpi=300 if suffix == "png" else None)
    plt.close(figure)
    return directory / "capacity.png"
