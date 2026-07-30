"""Failing publication-readiness specifications recorded before Round-4 fixes."""

from __future__ import annotations

import numpy as np
import pytest

from chronocline.channels import build_memoryless_channel, monte_carlo_matrix
from chronocline.channels.memoryless import build_random_phase_channel
from chronocline.config import RunConfig
from chronocline.distributions import laplace, student_t
from chronocline.exceptions import UnsupportedScientificModelError
from chronocline.experiments.runner import plan
from chronocline.quantization import UniformQuantizer
from chronocline.simulation import block_mutual_information, observation_trace


@pytest.mark.parametrize("jitter", [laplace(), student_t(5.0)])
def test_full_support_tail_intervals_are_positive_without_spurious_failure(jitter) -> None:
    """Tiny tail CDF/SF rounding differences must not abort full-support channels."""
    channel = build_memoryless_channel([0.0, 1.0], jitter, UniformQuantizer(0.5))
    assert np.all(channel.probabilities > 0)


def test_nearest_quantizer_matrix_matches_its_monte_carlo_definition() -> None:
    """Nearest labels and half-step analytical bins use the same quantizer convention."""
    quantizer = UniformQuantizer(0.5, phase=0.1, mode="nearest")
    channel = build_memoryless_channel([0.0, 1.0], laplace(), quantizer)
    empirical = monte_carlo_matrix(channel, laplace(), quantizer, 80_000, np.random.default_rng(3))
    assert np.max(np.abs(channel.probabilities - empirical)) < 0.015


def test_capacity_surface_plan_uses_cartesian_sweep_count(tmp_path) -> None:
    """A 2-by-2 surface has four scientific work units, not one runner default."""
    config = RunConfig.model_validate(
        {
            "experiment": {"kind": "capacity_surface", "name": "surface"},
            "channel": {"alphabet": {"values": [0.0, 1.0]}},
            "jitter": {"distribution": "gaussian"},
            "quantizer": {"step": 0.5},
            "sweep": {
                "parameters": {
                    "quantizer.step": [0.25, 0.5],
                    "channel.alphabet.values": [[0.0, 1.0], [0.0, 2.0]],
                }
            },
        }
    )
    assert plan(config).jobs == 4


def test_miller_madow_identity_correction_has_the_entropy_combination_sign() -> None:
    """For an identity trace the MI correction is negative, not the old opposite sign."""
    values = np.tile(np.array([0, 1]), 10)
    estimate = block_mutual_information(values, values, 1)
    assert estimate["miller_madow_block_mutual_information"] > estimate[
        "block_mutual_information_estimate"
    ]


def test_ambiguous_random_phase_channel_is_rejected() -> None:
    """Averaging phase-dependent DMCs is not an exact receiver semantics."""
    with pytest.raises(UnsupportedScientificModelError):
        build_random_phase_channel([0.0, 1.0], laplace(), 0.5)


def test_direct_batching_starts_from_arrivals_not_timestamp_quantization() -> None:
    """Direct release models preserve arrival resolution until their release rule."""
    delays = np.array([0.7, 0.7, 0.7])
    jitter = np.array([0.12, -0.11, 0.12])
    direct = observation_trace(delays, jitter, UniformQuantizer(0.5), model="arrival_timestamps")
    quantized = observation_trace(
        delays, jitter, UniformQuantizer(0.5), model="timestamp_quantization"
    )
    assert not np.array_equal(direct.observed_timestamps, quantized.observed_timestamps)
