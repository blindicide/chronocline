"""Detector summary and ROC plotting from stored source tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import require_variation, save_figure


def plot_detection(results: pd.DataFrame, directory: Path) -> Path:
    """Plot AUC against stored sample size."""
    data = results[results.metric_name == "auc"].copy()
    require_variation(data, "sample_size")
    figure, axis = plt.subplots(layout="constrained")
    axis.plot(data.sample_size, data.metric_value, marker="o")
    axis.set(xlabel="Observation count", ylabel="AUC")
    return save_figure(figure, directory, "auc_vs_sample_size", data)
