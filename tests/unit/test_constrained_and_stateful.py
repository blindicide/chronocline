import numpy as np
import pytest

from chronocline.information.constrained import constrained_capacity
from chronocline.quantization import UniformQuantizer
from chronocline.simulation import cumulative_timestamp_observations, empirical_mutual_information


def test_constrained_capacity_is_feasible() -> None:
    channel = np.array([[0.9, 0.1], [0.1, 0.9]])
    result = constrained_capacity(
        channel, np.array([0.0, 1.0]), np.array([0.5, 0.5]), max_mean_delay=0.6
    )
    assert result.feasible
    assert result.mean_delay <= 0.6 + 1e-8


def test_timestamp_preserve_order_and_empirical_mi() -> None:
    timestamps, delays = cumulative_timestamp_observations(
        np.array([1.0, 1.0]), np.array([0.0, 0.2, -0.8]), UniformQuantizer(0.5)
    )
    assert np.all(np.diff(timestamps) >= 0)
    assert np.all(delays >= 0)
    assert empirical_mutual_information(np.array([0, 1]), np.array([0, 1])) == pytest.approx(1.0)
