"""Strict, typed configuration and generic validated sweep resolution."""

from __future__ import annotations

from copy import deepcopy
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExperimentKind(StrEnum):
    """Scientific experiment identity used for runner and plotting dispatch."""

    SMOKE = "smoke"
    MEMORYLESS_BASELINE = "memoryless_baseline"
    CAPACITY_CURVE = "capacity_curve"
    CAPACITY_SURFACE = "capacity_surface"
    PHASE_SENSITIVITY = "phase_sensitivity"
    DETECTABILITY_FRONTIER = "detectability_frontier"
    FINITE_SAMPLE_DETECTION = "finite_sample_detection"
    ALPHABET_OPTIMIZATION = "alphabet_optimization"
    JITTER_COMPARISON = "jitter_comparison"
    BATCHING_COMPARISON = "batching_comparison"


class StrictModel(BaseModel):
    """Pydantic base model rejecting unknown keys and invalid assignments."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _finite(value: float, field: str) -> float:
    if not np.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


class ExperimentConfig(StrictModel):
    kind: ExperimentKind
    name: str
    seed: int = 20260728
    output_directory: str = "results"
    overwrite: bool = False
    locale: Literal["en", "ru"] = "en"
    workers: int = 1
    require_clean_git: bool = True

    @field_validator("seed")
    @classmethod
    def nonnegative_seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed must be non-negative")
        return value

    @field_validator("workers")
    @classmethod
    def positive_workers(cls, value: int) -> int:
        if value < 1:
            raise ValueError("workers must be at least one")
        return value

    @field_validator("output_directory")
    @classmethod
    def safe_relative_output(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("output_directory must be a safe relative path")
        return value


class AlphabetConfig(StrictModel):
    values: list[float]
    unit: str = "normalized"

    @field_validator("values")
    @classmethod
    def ordered(cls, value: list[float]) -> list[float]:
        if (
            not value
            or not all(np.isfinite(value))
            or any(b <= a for a, b in zip(value, value[1:], strict=False))
        ):
            raise ValueError("alphabet values must be finite and strictly ordered")
        return value


class ChannelConfig(StrictModel):
    mode: Literal["absolute_delay", "cover_perturbation"] = "absolute_delay"
    alphabet: AlphabetConfig
    input_probabilities: list[float] | None = None

    @model_validator(mode="after")
    def probabilities_match_alphabet(self) -> ChannelConfig:
        if self.input_probabilities is not None:
            p = np.asarray(self.input_probabilities, float)
            if len(p) != len(self.alphabet.values) or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
                raise ValueError("input_probabilities must match alphabet and sum to one")
        return self


class JitterConfig(StrictModel):
    distribution: Literal["gaussian", "laplace", "uniform", "student_t", "gaussian_mixture"]
    mean: float = 0.0
    scale: float = 1.0
    degrees_of_freedom: float | None = None
    lower: float | None = None
    upper: float | None = None
    weights: list[float] | None = None
    means: list[float] | None = None
    scales: list[float] | None = None

    @model_validator(mode="after")
    def valid_parameters(self) -> JitterConfig:
        _finite(self.mean, "mean")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("jitter scale must be positive and finite")
        if self.distribution == "uniform":
            if self.lower is None or self.upper is None or not self.lower < self.upper:
                raise ValueError("uniform jitter requires finite lower < upper")
        if self.distribution == "student_t":
            if self.degrees_of_freedom is None or self.degrees_of_freedom <= 0:
                raise ValueError("student_t requires positive degrees_of_freedom")
        if self.distribution == "gaussian_mixture":
            if self.weights is None or self.means is None or self.scales is None:
                raise ValueError("gaussian_mixture requires weights, means, and scales")
            w, m, s = map(np.asarray, (self.weights, self.means, self.scales))
            if (
                len(w) == 0
                or len(w) != len(m)
                or len(w) != len(s)
                or np.any(w < 0)
                or np.any(s <= 0)
                or not np.all(np.isfinite(np.r_[w, m, s]))
                or not np.isclose(w.sum(), 1.0)
            ):
                raise ValueError(
                    "mixture arrays must be finite, aligned, positive-scale, and weights sum to one"
                )
        return self


class QuantizerConfig(StrictModel):
    type: Literal["uniform"] = "uniform"
    step: float
    phase: float = 0.0
    mode: Literal["floor", "nearest"] = "floor"
    random_phase: Literal[False, "per_trace", "per_symbol_known", "per_symbol_unknown"] = False
    quadrature_points: int = 32

    @model_validator(mode="after")
    def valid_quantizer(self) -> QuantizerConfig:
        if not np.isfinite(self.step) or self.step <= 0 or not np.isfinite(self.phase):
            raise ValueError("quantizer step must be positive and phase finite")
        if self.quadrature_points < 2:
            raise ValueError("quadrature_points must be at least two")
        return self


class MatrixConfig(StrictModel):
    tail_probability: float = 1e-12
    tail_mode: Literal["overflow_bins", "conditional_truncation"] = "overflow_bins"

    @field_validator("tail_probability")
    @classmethod
    def valid_tail(cls, value: float) -> float:
        if not 0 < value < 1:
            raise ValueError("tail_probability must be in (0, 1)")
        return value


class OptimizationConfig(StrictModel):
    method: Literal["blahut_arimoto", "slsqp"] = "blahut_arimoto"
    tolerance: float = 1e-10
    max_iterations: int = 10000

    @model_validator(mode="after")
    def valid_optimization(self) -> OptimizationConfig:
        if self.tolerance <= 0 or self.max_iterations < 1:
            raise ValueError(
                "optimization tolerance must be positive and max_iterations at least one"
            )
        return self


class ConstraintsConfig(StrictModel):
    max_kl_bits: float | None = None
    max_mean_delay: float | None = None

    @model_validator(mode="after")
    def nonnegative_constraints(self) -> ConstraintsConfig:
        for value in (self.max_kl_bits, self.max_mean_delay):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError("constraints must be non-negative and finite")
        return self


class BaselineConfig(StrictModel):
    mode: Literal["channel_symbol", "input_distribution"] = "channel_symbol"
    symbol_index: int = 0
    input_probabilities: list[float] | None = None


class DetectionConfig(StrictModel):
    active_distribution_source: Literal[
        "configured_input", "unconstrained_capacity", "constrained_optimum"
    ] = "configured_input"
    sample_sizes: list[int] = Field(default_factory=lambda: [10, 25, 50, 100])
    trials: int = 300
    bootstrap_repetitions: int = 100
    target_false_positive_rates: list[float] = Field(default_factory=lambda: [0.01, 0.05, 0.1])
    batch_size: int = 256

    @model_validator(mode="after")
    def valid_detection(self) -> DetectionConfig:
        if (
            not self.sample_sizes
            or any(n < 1 for n in self.sample_sizes)
            or self.sample_sizes != sorted(set(self.sample_sizes))
        ):
            raise ValueError("sample_sizes must be positive and strictly increasing")
        if self.trials < 1 or self.bootstrap_repetitions < 1 or self.batch_size < 1:
            raise ValueError(
                "detector trials, bootstrap repetitions, and batch size must be positive"
            )
        if any(rate <= 0 or rate >= 1 for rate in self.target_false_positive_rates):
            raise ValueError("target false-positive rates must be in (0, 1)")
        return self


class SimulationConfig(StrictModel):
    trace_length: int = 2000
    replications: int = 3
    jitter_application: Literal["timestamp", "delay"] = "timestamp"
    preserve_order: bool = True
    block_lengths: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    discard_initial_transient: bool = True
    transient_observations: int = 1

    @model_validator(mode="after")
    def valid_simulation(self) -> SimulationConfig:
        if (
            self.trace_length < 2
            or self.replications < 1
            or not self.block_lengths
            or any(k < 1 for k in self.block_lengths)
            or self.transient_observations < 0
        ):
            raise ValueError("simulation trace, replications, and block lengths must be positive")
        return self


class BatchingConfig(StrictModel):
    modes: list[str] = Field(
        default_factory=lambda: [
            "no_batching",
            "timestamp_quantization",
            "fixed_window_observation",
            "ceiling_release",
        ]
    )
    windows: list[float] = Field(default_factory=lambda: [0.25, 0.5, 1.0])
    phase: float = 0.0
    maximum_batch_size: int | None = None

    @model_validator(mode="after")
    def valid_batching(self) -> BatchingConfig:
        if (
            not self.modes
            or not self.windows
            or any(w <= 0 or not np.isfinite(w) for w in self.windows)
        ):
            raise ValueError("batching modes and positive finite windows are required")
        if self.maximum_batch_size is not None and self.maximum_batch_size < 1:
            raise ValueError("maximum_batch_size must be positive")
        return self


class AlphabetSearchConfig(StrictModel):
    symbols: int = 2
    minimum: float = 0.0
    maximum: float = 4.0
    minimum_spacing: float = 0.05
    anchor_first_symbol: bool = True
    binary_grid_points: int = 101
    global_restarts: int = 4

    @model_validator(mode="after")
    def valid_search(self) -> AlphabetSearchConfig:
        if (
            self.symbols < 2
            or self.maximum <= self.minimum
            or self.minimum_spacing <= 0
            or self.maximum - self.minimum < self.minimum_spacing * (self.symbols - 1)
            or self.binary_grid_points < 3
        ):
            raise ValueError("invalid alphabet search bounds, spacing, or grid")
        return self


class SweepConfig(StrictModel):
    parameters: dict[str, list[Any]] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def nonempty_values(cls, value: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if any(not key or not values for key, values in value.items()):
            raise ValueError("sweep paths and value lists must be non-empty")
        return value


class RunConfig(StrictModel):
    experiment: ExperimentConfig
    channel: ChannelConfig
    jitter: JitterConfig
    quantizer: QuantizerConfig
    matrix: MatrixConfig = Field(default_factory=MatrixConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    constraints: ConstraintsConfig = Field(default_factory=ConstraintsConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    alphabet_search: AlphabetSearchConfig = Field(default_factory=AlphabetSearchConfig)
    sweep: SweepConfig = Field(default_factory=SweepConfig)

    @model_validator(mode="after")
    def baseline_valid(self) -> RunConfig:
        if self.baseline.mode == "channel_symbol" and not 0 <= self.baseline.symbol_index < len(
            self.channel.alphabet.values
        ):
            raise ValueError("baseline symbol_index must reference the input alphabet")
        if self.baseline.mode == "input_distribution":
            p = self.baseline.input_probabilities
            if (
                p is None
                or len(p) != len(self.channel.alphabet.values)
                or min(p) < 0
                or not np.isclose(sum(p), 1)
            ):
                raise ValueError("baseline input distribution must match alphabet and sum to one")
        return self


IMMUTABLE_SWEEP_PREFIXES = {
    "experiment.seed",
    "experiment.output_directory",
    "experiment.name",
    "experiment.kind",
    "experiment.require_clean_git",
}


def load_config(path: str | Path) -> RunConfig:
    """Load and strictly validate a YAML experiment configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        return RunConfig.model_validate(yaml.safe_load(handle))


def _set_path(data: dict[str, Any], path: str, value: Any) -> None:
    if path in IMMUTABLE_SWEEP_PREFIXES:
        raise ValueError(f"sweep cannot override provenance field {path}")
    cursor: dict[str, Any] = data
    parts = path.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"unknown sweep path {path}")
        cursor = child
    if parts[-1] not in cursor:
        raise ValueError(f"unknown sweep path {path}")
    cursor[parts[-1]] = value


def apply_overrides(config: RunConfig, overrides: dict[str, Any]) -> RunConfig:
    """Return a revalidated deep copy with every dotted override applied."""
    data = deepcopy(config.model_dump(mode="python"))
    for path, value in overrides.items():
        _set_path(data, path, value)
    return RunConfig.model_validate(data)
