"""Strict YAML experiment configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Pydantic model rejecting undeclared configuration keys."""

    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(StrictModel):
    name: str
    seed: int = 20260728
    output_directory: str = "results"
    overwrite: bool = False
    locale: Literal["en", "ru"] = "en"
    workers: int = 1

    @field_validator("seed")
    @classmethod
    def nonnegative_seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed must be non-negative")
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
        if len(value) < 1 or any(b <= a for a, b in zip(value, value[1:], strict=False)):
            raise ValueError("alphabet values must be strictly ordered")
        return value


class ChannelConfig(StrictModel):
    mode: Literal["absolute_delay", "cover_perturbation"] = "absolute_delay"
    alphabet: AlphabetConfig
    input_probabilities: list[float] | None = None

    @model_validator(mode="after")
    def probabilities_match_alphabet(self) -> ChannelConfig:
        if self.input_probabilities is not None:
            if len(self.input_probabilities) != len(self.alphabet.values):
                raise ValueError("input_probabilities length must match alphabet")
            if min(self.input_probabilities) < 0 or abs(sum(self.input_probabilities) - 1) > 1e-10:
                raise ValueError("input_probabilities must be non-negative and sum to one")
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
        if self.scale <= 0:
            raise ValueError("jitter scale must be positive")
        if self.distribution == "uniform" and (
            self.lower is None or self.upper is None or self.upper <= self.lower
        ):
            raise ValueError("uniform jitter requires lower < upper")
        if self.distribution == "student_t" and (
            self.degrees_of_freedom is None or self.degrees_of_freedom <= 0
        ):
            raise ValueError("student_t jitter requires positive degrees_of_freedom")
        return self


class QuantizerConfig(StrictModel):
    type: Literal["uniform"] = "uniform"
    step: float
    phase: float = 0.0
    mode: Literal["floor", "nearest"] = "floor"
    random_phase: Literal[False, "per_trace", "per_symbol"] = False
    quadrature_points: int = 32

    @field_validator("step")
    @classmethod
    def positive_step(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("quantizer step must be positive")
        return value


class MatrixConfig(StrictModel):
    tail_probability: float = 1e-12
    include_overflow_bins: bool = True

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


class ConstraintsConfig(StrictModel):
    max_kl_divergence: float | None = None
    max_mean_delay: float | None = None

    @model_validator(mode="after")
    def nonnegative_constraints(self) -> ConstraintsConfig:
        for value in (self.max_kl_divergence, self.max_mean_delay):
            if value is not None and value < 0:
                raise ValueError("constraints must be non-negative")
        return self


class SweepConfig(StrictModel):
    parameters: dict[str, list[Any]] = Field(default_factory=dict)


class RunConfig(StrictModel):
    experiment: ExperimentConfig
    channel: ChannelConfig
    jitter: JitterConfig
    quantizer: QuantizerConfig
    matrix: MatrixConfig = Field(default_factory=MatrixConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    constraints: ConstraintsConfig = Field(default_factory=ConstraintsConfig)
    sweep: SweepConfig = Field(default_factory=SweepConfig)


def load_config(path: str | Path) -> RunConfig:
    """Load and strictly validate a YAML experiment configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return RunConfig.model_validate(data)
