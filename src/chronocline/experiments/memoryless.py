"""Exact memoryless, curve, surface, and jitter-comparison experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..channels import build_memoryless_channel, monte_carlo_matrix
from ..config import ExperimentKind, RunConfig
from ..distributions import GaussianMixture, gaussian, laplace, student_t, uniform
from ..distributions.base import JitterDistribution
from ..information import blahut_arimoto, mutual_information
from ..quantization import UniformQuantizer
from .base import ExperimentContext, ExperimentOutput, ExperimentPlan


def make_jitter(config: RunConfig) -> JitterDistribution:
    """Construct the configured independent jitter distribution."""
    j = config.jitter
    if j.distribution == "gaussian":
        return gaussian(j.mean, j.scale)
    if j.distribution == "laplace":
        return laplace(j.mean, j.scale)
    if j.distribution == "uniform":
        if j.lower is None or j.upper is None:
            raise ValueError("uniform jitter bounds missing")
        return uniform(j.lower, j.upper)
    if j.distribution == "student_t":
        if j.degrees_of_freedom is None:
            raise ValueError("student t degrees missing")
        return student_t(j.degrees_of_freedom, j.mean, j.scale)
    return GaussianMixture(j.weights or [], j.means or [], j.scales or [])


def row(
    config: RunConfig,
    job: int,
    metric: str,
    value: float,
    units: str,
    **parameters: object,
) -> dict[str, object]:
    """Create one schema-2 scalar row."""
    job_id = hashlib.sha256(
        f"{config.experiment.kind}:{config.experiment.seed}:{job}:{parameters}".encode()
    ).hexdigest()[:12]
    estimator = str(parameters.pop("estimator", "exact"))
    return {
        "experiment_name": config.experiment.name,
        "experiment_kind": config.experiment.kind,
        "job_id": job_id,
        "sweep_index": job,
        "replication": None,
        "metric_name": metric,
        "metric_value": value,
        "units": units,
        "estimator": estimator,
        "status": "complete",
        **parameters,
    }


def evaluate_memoryless(
    config: RunConfig, job: int, directory: Path, *, monte_carlo: bool = False
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Evaluate one exact DMC and persist its matrix artifact."""
    quantizer = UniformQuantizer(
        config.quantizer.step, config.quantizer.phase, config.quantizer.mode
    )
    channel = build_memoryless_channel(
        config.channel.alphabet.values,
        make_jitter(config),
        quantizer,
        tail_probability=config.matrix.tail_probability,
        include_overflow_bins=config.matrix.tail_mode == "overflow_bins",
    )
    p = np.asarray(
        config.channel.input_probabilities or np.full(len(channel.inputs), 1 / len(channel.inputs))
    )
    capacity = blahut_arimoto(
        channel.probabilities,
        tolerance=config.optimization.tolerance,
        max_iterations=config.optimization.max_iterations,
    )
    matrix_path = directory / "matrices" / f"channel_{job}.npz"
    matrix_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        matrix_path,
        probabilities=channel.probabilities,
        inputs=channel.inputs,
        outputs=channel.outputs,
    )
    params: dict[str, object] = {
        "quantizer_step": config.quantizer.step,
        "quantizer_phase": config.quantizer.phase,
        "jitter_distribution": config.jitter.distribution,
        "jitter_scale": config.jitter.scale,
        "alphabet": jsonable(config.channel.alphabet.values),
    }
    rows = [
        row(
            config,
            job,
            "mutual_information",
            mutual_information(p, channel.probabilities),
            "bits_per_symbol",
            **params,
        ),
        row(
            config,
            job,
            "capacity_bits_per_symbol",
            capacity.capacity_bits,
            "bits_per_symbol",
            estimator="blahut_arimoto",
            **params,
        ),
        row(
            config,
            job,
            "capacity_residual",
            capacity.residual,
            "bits_per_symbol",
            estimator="blahut_arimoto",
            **params,
        ),
        row(config, job, "matrix_row_sum_error", channel.row_sum_error, "probability", **params),
    ]
    if monte_carlo:
        empirical = monte_carlo_matrix(
            channel,
            make_jitter(config),
            quantizer,
            20_000,
            np.random.default_rng(np.random.SeedSequence(config.experiment.seed, spawn_key=(job,))),
        )
        rows.append(
            row(
                config,
                job,
                "monte_carlo_max_absolute_error",
                float(np.max(np.abs(empirical - channel.probabilities))),
                "probability",
                estimator="monte_carlo",
                **params,
            )
        )
    return rows, capacity.input_probabilities


def jsonable(value: object) -> str:
    """Store structured parameter values in a stable tidy-table cell."""
    return str(value)


class MemorylessRunner:
    """Runner for smoke, baseline, capacity curve/surface, and jitter comparison."""

    def plan(self, config: RunConfig, output_directory: Path) -> ExperimentPlan:
        jobs = max(1, len(config.sweep.parameters.get("quantizer.step", [None])))
        return ExperimentPlan(
            config.experiment.kind,
            jobs,
            frozenset({"capacity_bits_per_symbol", "matrix_row_sum_error"}),
            frozenset({"matrices"}),
            output_directory,
        )

    def execute(
        self, context: ExperimentContext, jobs: list[tuple[int, RunConfig]]
    ) -> ExperimentOutput:
        rows: list[dict[str, object]] = []
        for index, config in jobs:
            variants = [config]
            if config.experiment.kind is ExperimentKind.JITTER_COMPARISON:
                variants = _matched_variance_jitter_variants(config)
            for variant_index, variant in enumerate(variants):
                job_index = index * len(variants) + variant_index
                job_rows, optimum = evaluate_memoryless(
                    variant,
                    job_index,
                    context.directory,
                    monte_carlo=variant.experiment.kind
                    in {ExperimentKind.SMOKE, ExperimentKind.MEMORYLESS_BASELINE},
                )
                variance = make_jitter(variant).variance()
                for entry in job_rows:
                    entry["jitter_variance"] = variance
                rows.extend(job_rows)
                if variant.experiment.kind is ExperimentKind.JITTER_COMPARISON:
                    rows.append(
                        row(
                            variant,
                            job_index,
                            "jitter_variance",
                            variance,
                            "normalized_time_squared",
                            estimator="analytic",
                            jitter_distribution=variant.jitter.distribution,
                            quantizer_step=variant.quantizer.step,
                        )
                    )
                for symbol, probability in enumerate(optimum):
                    rows.append(
                        row(
                            variant,
                            job_index,
                            "optimal_input_probability",
                            float(probability),
                            "probability",
                            symbol_index=symbol,
                            jitter_variance=variance,
                        )
                    )
        frame = pd.DataFrame(rows)
        (context.directory / "tables").mkdir(exist_ok=True)
        frame.to_csv(context.directory / "tables" / "figure_memoryless_source.csv", index=False)
        return ExperimentOutput(rows=rows, artifacts=[Path("tables/figure_memoryless_source.csv")])


def _matched_variance_jitter_variants(config: RunConfig) -> list[RunConfig]:
    """Return Gaussian, Laplace, t, and mixture laws with unit variance."""
    t_degrees = 5.0
    mixture_mean = 0.5
    mixture_scale = float(np.sqrt(1 - mixture_mean**2))
    specifications = [
        {"distribution": "gaussian", "scale": 1.0},
        {"distribution": "laplace", "scale": 1 / np.sqrt(2)},
        {
            "distribution": "student_t",
            "scale": np.sqrt((t_degrees - 2) / t_degrees),
            "degrees_of_freedom": t_degrees,
        },
        {
            "distribution": "gaussian_mixture",
            "scale": 1.0,
            "weights": [0.5, 0.5],
            "means": [-mixture_mean, mixture_mean],
            "scales": [mixture_scale, mixture_scale],
        },
    ]
    return [
        config.model_copy(update={"jitter": config.jitter.model_copy(update=specification)})
        for specification in specifications
    ]
