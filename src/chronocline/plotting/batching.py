"""Stateful batching comparison plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import require_variation, save_figure


def plot_batching(results: pd.DataFrame, directory: Path) -> Path:
    """Plot empirical symbol MI by stored batching mode/window."""
    data = results[results.metric_name == "symbol_mutual_information"].copy()
    require_variation(data, "batching_window")
    figure, axis = plt.subplots(layout="constrained")
    for mode, group in data.groupby("batching_mode"):
        axis.plot(group.batching_window, group.metric_value, marker="o", label=mode)
    axis.legend()
    axis.set(xlabel="Batching window", ylabel="Empirical symbol MI (bits/symbol)")
    return save_figure(figure, directory, "batching_information_comparison", data)
