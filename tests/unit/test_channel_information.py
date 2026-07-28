import numpy as np
import pytest

from chronocline.channels import build_memoryless_channel, monte_carlo_matrix
from chronocline.distributions import gaussian
from chronocline.information import blahut_arimoto, mutual_information, mutual_information_entropy
from chronocline.quantization import UniformQuantizer


@pytest.mark.parametrize("crossover", [0.01, 0.1, 0.25, 0.49])
def test_bsc_capacity(crossover: float) -> None:
    channel = np.array([[1 - crossover, crossover], [crossover, 1 - crossover]])
    expected = 1 + crossover * np.log2(crossover) + (1 - crossover) * np.log2(1 - crossover)
    result = blahut_arimoto(channel, tolerance=1e-12)
    assert result.capacity_bits == pytest.approx(expected, abs=1e-10)


def test_identical_rows_have_zero_capacity() -> None:
    result = blahut_arimoto(np.array([[0.3, 0.7], [0.3, 0.7]]))
    assert abs(result.capacity_bits) < 1e-10


def test_exact_matrix_is_stochastic_and_matches_mi_forms() -> None:
    channel = build_memoryless_channel([0.0, 1.0], gaussian(), UniformQuantizer(0.5))
    assert channel.row_sum_error < 1e-12
    p = np.array([0.5, 0.5])
    assert mutual_information(p, channel.probabilities) == pytest.approx(
        mutual_information_entropy(p, channel.probabilities)
    )


def test_monte_carlo_matrix_agrees_with_exact() -> None:
    channel = build_memoryless_channel([0.0, 1.0], gaussian(), UniformQuantizer(0.5))
    empirical = monte_carlo_matrix(
        channel, gaussian(), UniformQuantizer(0.5), 50_000, np.random.default_rng(5)
    )
    assert np.max(np.abs(empirical - channel.probabilities)) < 0.015
