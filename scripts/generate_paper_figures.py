"""Regenerate figures from a stored result directory."""

import sys

import pandas as pd

from chronocline.plotting import plot_capacity

if __name__ == "__main__":
    directory = sys.argv[1]
    plot_capacity(pd.read_csv(f"{directory}/results.csv"), f"{directory}/figures")
