import pytest

from chronocline.config import ExperimentKind, RunConfig, apply_overrides
from chronocline.experiments.runner import RUNNERS


def config(kind: ExperimentKind) -> RunConfig:
    return RunConfig.model_validate({"experiment": {"kind": kind, "name": "x", "require_clean_git": False}, "channel": {"alphabet": {"values": [0.0, 1.0]}}, "jitter": {"distribution": "gaussian"}, "quantizer": {"step": 0.5}})


def test_specialised_kinds_have_distinct_runners() -> None:
    assert type(RUNNERS[ExperimentKind.FINITE_SAMPLE_DETECTION]) is not type(RUNNERS[ExperimentKind.BATCHING_COMPARISON])
    assert type(RUNNERS[ExperimentKind.DETECTABILITY_FRONTIER]) is not type(RUNNERS[ExperimentKind.PHASE_SENSITIVITY])


def test_generic_override_revalidates_without_mutating_original() -> None:
    original = config(ExperimentKind.CAPACITY_CURVE)
    resolved = apply_overrides(original, {"quantizer.phase": 0.25, "jitter.scale": 2.0, "channel.alphabet.values": [0.0, 2.0]})
    assert original.quantizer.phase == 0.0
    assert resolved.quantizer.phase == 0.25
    assert resolved.jitter.scale == 2.0
    assert resolved.channel.alphabet.values == [0.0, 2.0]
    with pytest.raises(ValueError, match="unknown sweep path"):
        apply_overrides(original, {"unknown.path": 1})
