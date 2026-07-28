"""Small, real executions for each specialised experiment runner."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronocline.config import ExperimentKind, RunConfig
from chronocline.experiments.base import ExperimentContext
from chronocline.experiments.runner import RUNNERS
from chronocline.plotting import plot_result_directory
from chronocline.results.manifest import create_manifest, finalize_manifest
from chronocline.results.storage import write_results
from chronocline.results.validation import semantic_errors


def make_config(kind: ExperimentKind) -> RunConfig:
    """Return a computationally small valid config for one runner kind."""
    data: dict[str, object] = {
        "experiment": {
            "kind": kind,
            "name": kind.value,
            "require_clean_git": False,
        },
        "channel": {
            "alphabet": {"values": [0.0, 1.0]},
            "input_probabilities": [0.2, 0.8],
        },
        "jitter": {"distribution": "gaussian", "scale": 1.0},
        "quantizer": {"step": 0.5},
        "baseline": {"mode": "channel_symbol", "symbol_index": 0},
    }
    if kind is ExperimentKind.PHASE_SENSITIVITY:
        data["sweep"] = {"parameters": {"quantizer.phase": [0.0, 0.25]}}
    if kind is ExperimentKind.DETECTABILITY_FRONTIER:
        data["sweep"] = {"parameters": {"constraints.max_kl_bits": [0.01, 0.1]}}
        data["constraints"] = {"max_kl_bits": 0.01}
    if kind is ExperimentKind.BATCHING_COMPARISON:
        data["simulation"] = {
            "trace_length": 80,
            "replications": 1,
            "block_lengths": [1, 2],
        }
        data["batching"] = {
            "modes": ["no_batching", "ceiling_release"],
            "windows": [0.5, 1.0],
        }
    if kind is ExperimentKind.FINITE_SAMPLE_DETECTION:
        data["detection"] = {
            "sample_sizes": [5, 10],
            "trials": 20,
            "bootstrap_repetitions": 2,
            "target_false_positive_rates": [0.1],
        }
    if kind is ExperimentKind.CAPACITY_SURFACE:
        data["sweep"] = {
            "parameters": {
                "quantizer.step": [0.25, 0.5],
                "channel.alphabet.values": [[0.0, 0.5], [0.0, 1.0]],
            }
        }
    return RunConfig.model_validate(data)


def specialized_jobs(config: RunConfig) -> list[tuple[int, RunConfig]]:
    """Build the two explicit sweep jobs used by specialised direct tests."""
    kind = config.experiment.kind
    if kind is ExperimentKind.PHASE_SENSITIVITY:
        return [
            (
                index,
                config.model_copy(
                    update={"quantizer": config.quantizer.model_copy(update={"phase": phase})}
                ),
            )
            for index, phase in enumerate((0.0, 0.25))
        ]
    if kind is ExperimentKind.DETECTABILITY_FRONTIER:
        return [
            (
                index,
                config.model_copy(
                    update={
                        "constraints": config.constraints.model_copy(
                            update={"max_kl_bits": maximum}
                        )
                    }
                ),
            )
            for index, maximum in enumerate((0.01, 0.1))
        ]
    return [(0, config)]


@pytest.mark.parametrize(
    "kind",
    [
        ExperimentKind.PHASE_SENSITIVITY,
        ExperimentKind.DETECTABILITY_FRONTIER,
        ExperimentKind.FINITE_SAMPLE_DETECTION,
        ExperimentKind.BATCHING_COMPARISON,
        ExperimentKind.ALPHABET_OPTIMIZATION,
    ],
)
def test_specialized_runner_executes(kind: ExperimentKind, tmp_path: Path) -> None:
    """Each runner emits real, typed result rows for a minimal valid workload."""
    config = make_config(kind)
    context = ExperimentContext(
        config,
        tmp_path,
        np.random.SeedSequence(3),
        "abc",
        False,
    )

    output = RUNNERS[kind].execute(context, specialized_jobs(config))

    assert output.rows
    assert {entry["experiment_kind"] for entry in output.rows} == {kind.value}


def test_plot_dispatch_and_semantic_rejection(tmp_path: Path) -> None:
    """Plot dispatch uses manifest kind and validation rejects altered units."""
    config = make_config(ExperimentKind.FINITE_SAMPLE_DETECTION)
    context = ExperimentContext(
        config,
        tmp_path,
        np.random.SeedSequence(4),
        "abc",
        False,
    )
    output = RUNNERS[config.experiment.kind].execute(context, [(0, config)])
    write_results(tmp_path, output.rows)
    manifest = create_manifest(
        run_id="x",
        config_hash="h",
        experiment_name=config.experiment.name,
        experiment_kind=config.experiment.kind,
        runner_name="DetectionRunner",
        source_commit="abc",
        source_dirty=False,
        allow_dirty_override=False,
        workers=1,
        expected_jobs=1,
        expected_metrics=["auc"],
        locale="en",
    )
    finalize_manifest(tmp_path, manifest, 1, [])
    assert not semantic_errors(tmp_path)
    assert plot_result_directory(tmp_path).exists()
    finalize_manifest(tmp_path, manifest, 1, [])
    assert not semantic_errors(tmp_path)

    frame = pd.read_csv(tmp_path / "results.csv")
    frame.loc[:, "units"] = "wrong"
    frame.to_csv(tmp_path / "results.csv", index=False)
    assert semantic_errors(tmp_path)
