"""Public-module and mathematical edge-case coverage for scientific utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from chronocline.channels.cover_modulated import cover_perturbation_matrix
from chronocline.channels.stateful import StatefulChannelNotice
from chronocline.cli import app
from chronocline.detection import detection_trial, log_likelihood_ratio
from chronocline.detection.bootstrap import bootstrap_mean
from chronocline.detection.metrics import roc
from chronocline.distributions import (
    AR1Jitter,
    EmpiricalJitter,
    GaussianMixture,
    gaussian,
    laplace,
    student_t,
    uniform,
)
from chronocline.information import bounds
from chronocline.information.divergence import (
    bhattacharyya,
    chi_square,
    hellinger,
    js_divergence,
    kl_divergence,
    total_variation,
)
from chronocline.logging_utils import configure_logging
from chronocline.optimization import best_found_alphabet, binary_grid_optimize, simplex_grid
from chronocline.optimization.diagnostics import ordered
from chronocline.optimization.pareto import nondominated
from chronocline.pcap import read_timestamps
from chronocline.quantization import UniformQuantizer
from chronocline.quantization.random_phase import phase_nodes
from chronocline.random import generator
from chronocline.simulation.batching import ceiling_release, fixed_window
from chronocline.simulation.sequence import symbols_to_delays


def test_detector_metrics_bootstrap_and_support_edges() -> None:
    """Finite detector primitives cover finite and infinite support decisions."""
    active = np.array([0.8, 0.2])
    baseline = np.array([0.2, 0.8])
    trial = detection_trial(active, baseline, n=8, trials=20, rng=np.random.default_rng(1))
    assert 0.5 <= float(trial["auc"]) <= 1
    assert 0 <= float(trial["minimum_total_error"]) <= 0.5
    assert log_likelihood_ratio(np.array([0]), np.array([0.0, 1.0]), baseline) == -np.inf
    assert log_likelihood_ratio(np.array([0]), active, np.array([0.0, 1.0])) == np.inf
    fpr, tpr, auc = roc(np.array([0.0, 0.1]), np.array([0.2, 0.3]))
    assert fpr[0] == tpr[0] == 0 and auc == pytest.approx(1.0)
    low, high = bootstrap_mean(np.array([1.0, 2.0, 3.0]), 30, np.random.default_rng(2))
    assert low <= high


def test_distribution_models_file_provenance_and_validation(tmp_path: Path) -> None:
    """Analytic, empirical, mixture, and stateful jitter remain well defined."""
    values = np.array([-1.0, 0.0, 2.0])
    empirical = EmpiricalJitter(values)
    assert empirical.cdf(np.array([-2.0, 0.0, 3.0])).tolist() == [0.0, 2 / 3, 1.0]
    assert empirical.ppf([0.0, 1.0]).tolist() == [-1.0, 2.0]
    assert empirical.support() == (-1.0, 2.0)
    assert empirical.sample(4, np.random.default_rng(3)).shape == (4,)
    assert empirical.mean() == pytest.approx(1 / 3)
    assert empirical.variance() > 0
    with pytest.raises(ValueError, match="quantiles"):
        empirical.ppf([-0.1])
    csv_path = tmp_path / "samples.csv"
    pd.DataFrame({"jitter": values}).to_csv(csv_path, index=False)
    from_file = EmpiricalJitter.from_file(csv_path, "jitter")
    assert from_file.source_hash and from_file.samples.tolist() == values.tolist()

    mixture = GaussianMixture([0.25, 0.75], [-1.0, 1.0], [0.5, 1.0])
    assert mixture.pdf(0.0) > 0 and 0 < mixture.cdf(0.0) < 1
    assert mixture.ppf([0.5]).shape == (1,)
    assert mixture.sample((2, 3), np.random.default_rng(4)).shape == (2, 3)
    assert mixture.support() == (-np.inf, np.inf)
    assert mixture.variance() > 0
    with pytest.raises(ValueError, match="weights"):
        GaussianMixture([0.0], [0.0], [1.0])
    with pytest.raises(ValueError, match="quantiles"):
        mixture.ppf([1.1])

    for law in (gaussian(), laplace(), uniform(-1, 1), student_t(4)):
        assert law.pdf(0.0) >= 0
        assert law.sample(3, np.random.default_rng(5)).shape == (3,)
        assert law.support()[0] <= law.mean() <= law.support()[1]
    with pytest.raises(ValueError):
        uniform(1, 1)
    with pytest.raises(ValueError):
        student_t(0)
    ar1 = AR1Jitter(0.8, mean_value=1.0, variance_value=2.0)
    assert ar1.sample(8, np.random.default_rng(6)).shape == (8,)
    with pytest.raises(ValueError):
        AR1Jitter(1.0)
    with pytest.raises(ValueError):
        ar1.sample(0, np.random.default_rng(7))


def test_information_divergences_and_optimization_edges() -> None:
    """Known distribution distances and bounded optimisation helpers agree."""
    p = np.array([0.5, 0.5])
    q = np.array([1.0, 0.0])
    assert kl_divergence(p, q) == np.inf
    assert chi_square(p, q) == np.inf
    assert total_variation(p, q) == pytest.approx(0.5)
    assert 0 < js_divergence(p, q) < 1
    assert 0 < hellinger(p, q) <= 1
    assert 0 < bhattacharyya(p, q) < 1
    assert bounds.pinsker_upper_bound(1.0) > 0

    exact = binary_grid_optimize(0, 1, 4, lambda alphabet: alphabet[1] - alphabet[0])
    assert exact.label == "exact_grid_optimum" and exact.objective == pytest.approx(1)
    found = best_found_alphabet(3, 0, 3, 0.5, np.sum, seed=8)
    assert found.label == "best_found_numerical_solution"
    assert ordered(found.alphabet, 0.5)
    assert simplex_grid(2, 2).shape == (3, 2)
    assert simplex_grid(3, 2).shape == (6, 3)
    with pytest.raises(ValueError):
        simplex_grid(4, 2)
    frontier = nondominated(np.array([[1.0, 2.0], [2.0, 2.0], [1.0, 1.0]]))
    assert {tuple(point) for point in frontier} == {(2.0, 2.0), (1.0, 1.0)}


def test_quantization_channel_helpers_random_streams_and_optional_pcap(tmp_path: Path) -> None:
    """Low-level stateful and sampled utilities preserve their published semantics."""
    nodes, weights = phase_nodes(2.0, 4)
    assert np.all((0 <= nodes) & (nodes <= 2.0)) and weights.sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        phase_nodes(0.0)
    nearest = UniformQuantizer(1.0, phase=1.25, mode="nearest")
    assert nearest.phase == pytest.approx(0.25)
    assert nearest.boundaries(0) == pytest.approx((-0.25, 0.75))
    assert nearest.metadata()["mode"] == "nearest"
    with pytest.raises(ValueError):
        UniformQuantizer(0)
    assert np.array_equal(symbols_to_delays(np.array([1, 0]), np.array([2.0, 3.0])), [3.0, 2.0])
    timestamps = np.array([0.1, 0.9])
    assert np.array_equal(fixed_window(timestamps, 1.0), [0.0, 0.0])
    assert np.array_equal(ceiling_release(timestamps, 1.0), [1.0, 1.0])
    assert StatefulChannelNotice("plugin").stateful
    first = generator(9, 1).normal(size=3)
    assert np.array_equal(first, generator(9, 1).normal(size=3))
    with pytest.raises(ValueError):
        generator(-1)

    channel = cover_perturbation_matrix(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        lambda count, rng: rng.normal(0, 0.1, count),
        UniformQuantizer(0.5),
        np.random.default_rng(10),
        samples=1000,
    )
    assert channel.metadata["construction"] == "monte_carlo"
    assert channel.row_sum_error < 1e-12
    with pytest.raises((RuntimeError, FileNotFoundError)):
        read_timestamps(tmp_path / "missing.pcap")


def test_cli_dispatch_validation_and_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI commands validate their kind contracts before invoking workloads."""
    runner = CliRunner()
    smoke = Path("configs/smoke.yaml")
    assert runner.invoke(app, ["validate-config", str(smoke)]).exit_code == 0
    assert runner.invoke(app, ["experiment", str(smoke), "--dry-run"]).exit_code == 0
    batching = tmp_path / "batching.yaml"
    batching.write_text(
        "experiment:\n"
        "  kind: batching_comparison\n"
        "  name: x\n"
        "  require_clean_git: false\n"
        "channel:\n"
        "  alphabet:\n"
        "    values: [0.0, 1.0]\n"
        "jitter:\n"
        "  distribution: gaussian\n"
        "quantizer:\n"
        "  step: 0.5\n"
    )
    result = runner.invoke(app, ["matrix", str(batching)])
    assert result.exit_code != 0 and "memoryless-compatible" in result.output
    result = runner.invoke(app, ["detect", str(smoke)])
    assert result.exit_code != 0 and "finite_sample_detection" in result.output
    captured: dict[str, object] = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: captured.update(kwargs))
    configure_logging(verbose=True)
    assert captured["level"] == logging.DEBUG
