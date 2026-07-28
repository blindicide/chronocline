import numpy as np
import pytest

from chronocline.information.constrained import constrained_capacity
from chronocline.quantization import UniformQuantizer
from chronocline.simulation import cumulative_timestamp_observations, empirical_mutual_information


def test_constrained_capacity_is_feasible() -> None:
    channel = np.array([[0.9, 0.1], [0.1, 0.9]])
    result = constrained_capacity(
        channel,
        np.array([0.0, 1.0]),
        np.array([0.5, 0.5]),
        max_mean_delay=0.6,
    )
    assert result.feasible
    assert result.mean_delay <= 0.6 + 1e-8


def test_binary_kl_grid_is_nontrivial_and_monotone() -> None:
    """The exact binary fallback traces an expanding feasible set by KL budget."""
    channel = np.array([[0.9, 0.1], [0.1, 0.9]])
    restrictive = constrained_capacity(
        channel,
        np.array([0.0, 1.0]),
        channel[0],
        max_kl_bits=0.01,
    )
    relaxed = constrained_capacity(
        channel,
        np.array([0.0, 1.0]),
        channel[0],
        max_kl_bits=0.5,
    )
    assert restrictive.converged and restrictive.kl_divergence_bits <= 0.01 + 1e-8
    assert relaxed.capacity_bits >= restrictive.capacity_bits


def test_multisymbol_slsqp_respects_configured_iteration_limit() -> None:
    """The generic multi-symbol solver remains available beyond binary grid cases."""
    channel = np.eye(3)
    result = constrained_capacity(
        channel,
        np.array([0.0, 0.5, 1.0]),
        np.full(3, 1 / 3),
        max_mean_delay=0.8,
        max_iterations=100,
        starts=4,
    )
    assert result.feasible and result.iterations <= 100


def test_timestamp_preserve_order_and_empirical_mi() -> None:
    timestamps, delays = cumulative_timestamp_observations(
        np.array([1.0, 1.0]), np.array([0.0, 0.2, -0.8]), UniformQuantizer(0.5)
    )
    assert np.all(np.diff(timestamps) >= 0)
    assert np.all(delays >= 0)
    assert empirical_mutual_information(np.array([0, 1]), np.array([0, 1])) == pytest.approx(1.0)
