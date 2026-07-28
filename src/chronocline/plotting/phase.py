"""Quantizer phase sensitivity plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import require_variation, save_figure


def plot_phase(results: pd.DataFrame, directory: Path) -> Path:
    """Plot capacity against explicit quantizer phase."""
    data = results[results.metric_name == "capacity_bits_per_symbol"].copy()
    require_variation(data, "quantizer_phase")
    figure, axis = plt.subplots(layout="constrained")
    axis.plot(data.quantizer_phase, data.metric_value, marker="o")
    axis.set(xlabel="Quantizer phase φ", ylabel="Capacity (bits/symbol)")
    return save_figure(figure, directory, "capacity_vs_phase", data)
