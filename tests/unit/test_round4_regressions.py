"""Failing publication-readiness specifications recorded before Round-4 fixes."""

from __future__ import annotations

import numpy as np
import pytest

from chronocline.channels import build_memoryless_channel, monte_carlo_matrix
from chronocline.config import RunConfig
from chronocline.distributions import laplace, student_t
from chronocline.experiments.runner import plan
from chronocline.quantization import UniformQuantizer


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
