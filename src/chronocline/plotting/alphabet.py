"""Optimised alphabet geometry plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import save_figure


def plot_alphabet(results: pd.DataFrame, directory: Path) -> Path:
    """Plot stored optimised symbol values without recomputation."""
    data = results[results.metric_name == "optimized_alphabet"].copy()
    figure, axis = plt.subplots(layout="constrained")
    axis.scatter(data.symbol_index, data.metric_value)
    axis.set(xlabel="Symbol index", ylabel="Optimized delay")
    return save_figure(figure, directory, "optimized_alphabet", data)
