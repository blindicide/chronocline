"""Capacity-detectability frontier plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import require_variation, save_figure


def plot_frontier(results: pd.DataFrame, directory: Path) -> Path:
    """Plot constrained capacity against achieved KL."""
    capacity = results[results.metric_name == "constrained_capacity_bits_per_symbol"]
    kl = results[results.metric_name == "achieved_kl_bits"]
    data = capacity.merge(
        kl[["job_id", "metric_value"]],
        on="job_id",
        suffixes=("_capacity", "_kl"),
    )
    require_variation(data, "metric_value_kl")
    figure, axis = plt.subplots(layout="constrained")
    axis.plot(data.metric_value_kl, data.metric_value_capacity, marker="o")
    axis.set(xlabel="Achieved KL (bits)", ylabel="Constrained capacity (bits/symbol)")
    return save_figure(figure, directory, "capacity_detectability_frontier", data)
