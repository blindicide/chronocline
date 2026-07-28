"""Regression specifications for the third scientific-correctness pass.

These tests intentionally describe behavior absent from the v0.6 baseline.  They
are committed before the corresponding implementation changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from chronocline.channels import build_memoryless_channel, monte_carlo_matrix
from chronocline.cli import app
from chronocline.config import RunConfig
from chronocline.distributions import gaussian
from chronocline.experiments.base import ExperimentContext
from chronocline.experiments.batching import BatchingRunner
from chronocline.experiments.detection import DetectionRunner
from chronocline.information.divergence import kl_divergence
from chronocline.optimization import best_found_alphabet
from chronocline.quantization import UniformQuantizer
from chronocline.results.validation import semantic_errors
from chronocline.simulation import batch_observation_trace, observation_trace


def detector_config() -> RunConfig:
    """Use a small, nontrivial finite-sample detector configuration."""
    return RunConfig.model_validate(
        {
            "experiment": {
                "kind": "finite_sample_detection",
                "name": "round3_detector",
                "require_clean_git": False,
            },
            "channel": {
                "alphabet": {"values": [0.0, 1.0]},
                "input_probabilities": [0.2, 0.8],
            },
            "jitter": {"distribution": "gaussian", "scale": 1.0},
            "quantizer": {"step": 0.5},
            "baseline": {"mode": "input_distribution", "input_probabilities": [0.99, 0.01]},
            "detection": {
                "sample_sizes": [5, 10],
                "trials": 20,
                "bootstrap_repetitions": 4,
                "target_false_positive_rates": [0.1],
                "batch_size": 8,
            },
        }
    )


def test_gaussian_far_upper_tail_is_positive_and_kl_is_finite() -> None:
    """Full-support Gaussian rows must not acquire artificial zero tail support."""
    jitter = gaussian()
    assert float(jitter.sf(9.0)) > 0.0
    channel = build_memoryless_channel([0.0, 1.0], jitter, UniformQuantizer(0.5))
    assert np.all(channel.probabilities > 0)
    assert np.isfinite(kl_divergence(channel.probabilities[1], channel.probabilities[0]))


def test_conditional_truncation_monte_carlo_never_requires_overflow_labels() -> None:
    """Conditional support is sampled conditionally instead of indexing absent tail bins."""
    channel = build_memoryless_channel(
        [0.0, 1.0],
        gaussian(),
        UniformQuantizer(0.5),
        include_overflow_bins=False,
    )
    empirical = monte_carlo_matrix(
        channel,
        gaussian(),
        UniformQuantizer(0.5),
        5_000,
        np.random.default_rng(7),
    )
    assert empirical.shape == channel.probabilities.shape
    assert np.allclose(empirical.sum(axis=1), 1.0)


def test_detector_has_one_canonical_tpr_row_per_scientific_key(tmp_path: Path) -> None:
    """TPR rows are unique by sample size, target FPR, hypothesis pair, and replication."""
    config = detector_config()
    output = DetectionRunner().execute(
        ExperimentContext(config, tmp_path, np.random.SeedSequence(9), "test", False),
        [(0, config)],
    )
    tpr = [row for row in output.rows if row["metric_name"] == "tpr_at_fpr"]
    keys = [
        (
            row["sample_size"],
            row.get("target_false_positive_rate"),
            row.get("hypothesis_pair"),
            row.get("replication"),
        )
        for row in tpr
    ]
    assert len(keys) == len(set(keys))
    assert all("target_fpr" not in row for row in tpr)


def test_detector_work_units_count_sample_sizes_and_hypothesis_pairs(tmp_path: Path) -> None:
    """Planning counts real detector work, including the required null control."""
    config = detector_config()
    plan = DetectionRunner().plan(config, tmp_path)
    assert plan.jobs == len(config.detection.sample_sizes) * 2


def test_detector_emits_a_null_control_with_chance_level_auc(tmp_path: Path) -> None:
    """A baseline-vs-baseline detector is a true null experiment, not omitted metadata."""
    config = detector_config()
    output = DetectionRunner().execute(
        ExperimentContext(config, tmp_path, np.random.SeedSequence(19), "test", False),
        [(0, config)],
    )
    null_auc = [
        row["metric_value"]
        for row in output.rows
        if row["metric_name"] == "auc" and row["hypothesis_pair"] == "baseline_vs_baseline"
    ]
    assert null_auc and np.allclose(null_auc, 0.5)


def test_ternary_search_can_leave_unused_upper_range_slack() -> None:
    """A ternary optimizer must not force the final symbol to the configured maximum."""
    result = best_found_alphabet(
        3,
        0.0,
        10.0,
        0.5,
        lambda alphabet: -float((alphabet[-1] - 2.0) ** 2),
        seed=3,
    )
    assert result.alphabet[-1] < 9.0


def test_ternary_restarts_and_unanchored_lower_slack_are_real_search_dimensions() -> None:
    """Restart metadata and unanchored first-symbol slack are not decorative fields."""
    result = best_found_alphabet(
        3,
        0.0,
        10.0,
        0.5,
        lambda alphabet: -float((alphabet[0] - 2.0) ** 2),
        seed=7,
        anchor_first_symbol=False,
        global_restarts=3,
    )
    assert result.alphabet[0] > 1.0
    assert len(result.restarts) == 3
    assert result.label == "ternary_differential_evolution_best_found"


def test_plotting_never_converts_failed_computation_to_passed(tmp_path: Path) -> None:
    """Adding a figure may add checksums but cannot erase failed semantic status."""
    pd.DataFrame(
        [
            {
                "metric_name": "capacity_bits_per_symbol",
                "metric_value": 0.2,
                "quantizer_step": 0.25,
            },
            {
                "metric_name": "capacity_bits_per_symbol",
                "metric_value": 0.3,
                "quantizer_step": 0.5,
            },
        ]
    ).to_csv(tmp_path / "results.csv", index=False)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_kind": "capacity_curve",
                "completion_status": "failed",
                "semantic_validation_status": "failed",
                "semantic_validation_errors": ["prior computation failure"],
                "completed_jobs": 0,
                "generated_files": {},
            }
        )
    )
    result = CliRunner().invoke(app, ["plot", str(tmp_path)])
    assert result.exit_code != 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["semantic_validation_status"] == "failed"


def test_semantic_validation_rejects_duplicate_scientific_rows(tmp_path: Path) -> None:
    """Duplicate scalar rows cannot silently pass structural validation."""
    rows = pd.DataFrame(
        [
            {
                "experiment_name": "x",
                "experiment_kind": "finite_sample_detection",
                "job_id": "same",
                "sweep_index": 0,
                "metric_name": "auc",
                "metric_value": 0.6,
                "units": "dimensionless",
                "estimator": "likelihood_ratio",
                "status": "complete",
                "sample_size": 10,
            },
            {
                "experiment_name": "x",
                "experiment_kind": "finite_sample_detection",
                "job_id": "same",
                "sweep_index": 0,
                "metric_name": "auc",
                "metric_value": 0.7,
                "units": "dimensionless",
                "estimator": "likelihood_ratio",
                "status": "complete",
                "sample_size": 10,
            },
        ]
    )
    rows.to_csv(tmp_path / "results.csv", index=False)
    (tmp_path / "tables").mkdir()
    pd.DataFrame({"false_positive_rate": [0.0, 1.0], "true_positive_rate": [0.0, 1.0]}).to_csv(
        tmp_path / "tables" / "roc_n_10.csv", index=False
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_kind": "finite_sample_detection",
                "completion_status": "complete",
                "result_schema_version": "2.0",
                "config_hash": "x",
                "source_commit": "x",
                "source_dirty": False,
                "expected_jobs": 1,
                "completed_jobs": 1,
                "generated_files": {},
            }
        )
    )
    assert any("duplicate" in error for error in semantic_errors(tmp_path))


@pytest.mark.parametrize("mode", ["per_trace", "per_symbol_unknown"])
def test_ambiguous_random_phase_modes_are_rejected(mode: str) -> None:
    """Unsupported random-phase semantics cannot be silently averaged into a DMC."""
    with pytest.raises(NotImplementedError):
        build_memoryless_channel(
            [0.0, 1.0],
            gaussian(),
            UniformQuantizer(0.5),
            random_phase_mode=mode,
        )


def test_delay_and_timestamp_models_have_distinct_explicit_trace_stages() -> None:
    """Delay noise and timestamp noise are not interchangeable simulation labels."""
    delays = np.array([1.0, 1.0, 1.0])
    jitter = np.array([0.4, -0.4, 0.4])
    quantizer = UniformQuantizer(0.5)
    delay_trace = observation_trace(delays, jitter, quantizer, model="delay_quantization")
    timestamp_trace = observation_trace(delays, jitter, quantizer, model="timestamp_quantization")
    assert not np.array_equal(delay_trace.observed_delays, timestamp_trace.observed_delays)
    batched = batch_observation_trace(timestamp_trace, 1.0, maximum_batch_size=1)
    assert np.array_equal(batched.packet_ids, np.arange(3))
    assert batched.batch_ids.tolist() == [0, 1, 2]


def test_batching_executes_replications_and_keeps_unbatched_packets_singleton(
    tmp_path: Path,
) -> None:
    """Configured replications and no-batching semantics affect every emitted row."""
    raw = detector_config().model_dump(mode="json")
    raw["experiment"].update({"kind": "batching_comparison", "name": "round3_batching"})
    raw["simulation"] = {"trace_length": 40, "replications": 2, "block_lengths": [1]}
    raw["batching"] = {"modes": ["ideal_delays", "no_batching"], "windows": [0.5]}
    config = RunConfig.model_validate(raw)
    output = BatchingRunner().execute(
        ExperimentContext(config, tmp_path, np.random.SeedSequence(11), "test", False),
        [(0, config)],
    )
    sizes = [
        row
        for row in output.rows
        if row["metric_name"] == "batch_size_mean" and row["batching_mode"] == "no_batching"
    ]
    assert len(sizes) == 2 and all(row["metric_value"] == 1 for row in sizes)
    assert {row["replication"] for row in sizes} == {0, 1}
