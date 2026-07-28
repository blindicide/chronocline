"""Synthetic selected-symbol timing sequence generation."""

import numpy as np


def symbols_to_delays(symbols: np.ndarray, alphabet: np.ndarray) -> np.ndarray:
    """Map integer symbol sequence to selected delay values."""
    return np.asarray(alphabet, float)[np.asarray(symbols, int)]
