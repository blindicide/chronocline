"""Capacity curves and two-dimensional surface figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import require_variation, save_figure
from .labels import label


def plot_capacity(results: pd.DataFrame, directory: str | Path, locale: str = "en") -> Path:
    """Plot capacity against the stored quantizer step only."""
    data = results[results.metric_name == "capacity_bits_per_symbol"].copy()
    require_variation(data, "quantizer_step")
    figure, axis = plt.subplots(layout="constrained")
    axis.plot(data.quantizer_step, data.metric_value, marker="o")
    axis.set(xlabel=label("step", locale), ylabel=label("capacity", locale))
    return save_figure(figure, Path(directory), "capacity_vs_quantizer_step", data)


def plot_capacity_surface(results: pd.DataFrame, directory: str | Path, locale: str = "en") -> Path:
    """Plot a true two-axis capacity scatter/heatmap source."""
    data = results[results.metric_name == "capacity_bits_per_symbol"].copy()
    require_variation(data, "quantizer_step")
    require_variation(data, "alphabet")
    figure, axis = plt.subplots(layout="constrained")
    image = axis.scatter(
        data.quantizer_step,
        data.alphabet.astype("category").cat.codes,
        c=data.metric_value,
        cmap="viridis",
    )
    axis.set(xlabel=label("step", locale), ylabel="Alphabet geometry")
    figure.colorbar(image, ax=axis, label=label("capacity", locale))
    return save_figure(figure, Path(directory), "capacity_surface_step_spacing", data)
