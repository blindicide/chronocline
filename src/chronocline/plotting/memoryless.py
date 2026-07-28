"""Memoryless baseline plotting using stored channel summary rows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import save_figure


def plot_memoryless_summary(results: pd.DataFrame, directory: Path) -> Path:
    """Plot exact MI and capacity as named categorical scientific quantities."""
    data = results[results.metric_name.isin(["mutual_information", "capacity_bits_per_symbol"])]
    if data.empty:
        raise ValueError("memoryless summary requires information metrics")
    values = data.groupby("metric_name", as_index=False).metric_value.mean()
    figure, axis = plt.subplots(layout="constrained")
    axis.bar(values.metric_name.str.replace("_", " "), values.metric_value)
    axis.set(xlabel="Information metric", ylabel="Bits per symbol")
    axis.tick_params(axis="x", rotation=20)
    return save_figure(figure, directory, "memoryless_information_summary", values)
