"""Jitter comparison plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import require_variation, save_figure


def plot_jitter(results: pd.DataFrame, directory: Path) -> Path:
    """Plot capacity curves grouped by stored jitter family."""
    data = results[results.metric_name == "capacity_bits_per_symbol"].copy()
    require_variation(data, "quantizer_step")
    figure, axis = plt.subplots(layout="constrained")
    for family, group in data.groupby("jitter_distribution"):
        axis.plot(group.quantizer_step, group.metric_value, marker="o", label=family)
    axis.legend()
    axis.set(xlabel="Quantizer step", ylabel="Capacity (bits/symbol)")
    return save_figure(figure, directory, "capacity_by_jitter_family", data)
