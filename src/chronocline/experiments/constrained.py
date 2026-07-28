"""Detectability-constrained capacity frontier experiment."""

from __future__ import annotations

import numpy as np

from ..channels import build_memoryless_channel
from ..config import RunConfig
from ..information import mutual_information
from ..information.constrained import constrained_capacity
from ..information.divergence import kl_divergence
from ..quantization import UniformQuantizer
from .base import ExperimentContext, ExperimentOutput, ExperimentPlan
from .memoryless import make_jitter, row


def baseline_output(config: RunConfig, matrix: np.ndarray) -> np.ndarray:
    """Construct the baseline on exactly the active channel output support."""
    if config.baseline.mode == "channel_symbol":
        return matrix[config.baseline.symbol_index]
    probabilities = np.asarray(config.baseline.input_probabilities, float)
    return probabilities @ matrix


class ConstrainedRunner:
    """Solve every configured KL budget against a distinct baseline output."""

    def plan(self, config: RunConfig, output_directory):
        return ExperimentPlan(
            config.experiment.kind,
            len(config.sweep.parameters.get("constraints.max_kl_bits", [])),
            frozenset(
                {
                    "constrained_capacity_bits_per_symbol",
                    "achieved_kl_bits",
                    "optimizer_converged",
                    "optimizer_feasible",
                }
            ),
            frozenset({"tables/figure_capacity_detectability_frontier.csv"}),
            output_directory,
        )

    def execute(
        self, context: ExperimentContext, jobs: list[tuple[int, RunConfig]]
    ) -> ExperimentOutput:
        rows = []
        previous = -np.inf
        for index, config in jobs:
            q = UniformQuantizer(
                config.quantizer.step, config.quantizer.phase, config.quantizer.mode
            )
            matrix = build_memoryless_channel(
                config.channel.alphabet.values,
                make_jitter(config),
                q,
                tail_probability=config.matrix.tail_probability,
                include_overflow_bins=config.matrix.tail_mode == "overflow_bins",
            )
            baseline = baseline_output(config, matrix.probabilities)
            result = constrained_capacity(
                matrix.probabilities,
                matrix.inputs,
                baseline,
                max_kl_bits=config.constraints.max_kl_bits,
                max_mean_delay=config.constraints.max_mean_delay,
                tolerance=config.optimization.tolerance,
                max_iterations=config.optimization.max_iterations,
                starts=16,
                seed=config.experiment.seed + index,
            )
            accepted = result.feasible and result.converged and np.isfinite(result.capacity_bits)
            if accepted and result.capacity_bits < previous - 1e-7:
                accepted = False
            if accepted:
                previous = result.capacity_bits
            params: dict[str, object] = {
                "max_kl_bits": config.constraints.max_kl_bits,
                "baseline_mode": config.baseline.mode,
            }
            grid = np.linspace(0.0, 1.0, 1001)
            grid_feasible: list[float] = []
            for first_probability in grid:
                probabilities = np.array([first_probability, 1 - first_probability])
                output = probabilities @ matrix.probabilities
                if (
                    config.constraints.max_kl_bits is None
                    or kl_divergence(output, baseline)
                    <= config.constraints.max_kl_bits + config.optimization.tolerance
                ) and (
                    config.constraints.max_mean_delay is None
                    or probabilities @ matrix.inputs
                    <= config.constraints.max_mean_delay + config.optimization.tolerance
                ):
                    grid_feasible.append(mutual_information(probabilities, matrix.probabilities))
            grid_best = max(grid_feasible, default=float("nan"))
            rows.extend(
                [
                    row(
                        config,
                        index,
                        "constrained_capacity_bits_per_symbol",
                        result.capacity_bits,
                        "bits_per_symbol",
                        estimator="slsqp",
                        **params,
                    ),
                    row(
                        config,
                        index,
                        "achieved_kl_bits",
                        result.kl_divergence_bits,
                        "bits",
                        estimator="slsqp",
                        **params,
                    ),
                    row(
                        config,
                        index,
                        "mean_delay",
                        result.mean_delay,
                        "normalized_time",
                        estimator="slsqp",
                        **params,
                    ),
                    row(
                        config,
                        index,
                        "optimizer_converged",
                        float(result.converged),
                        "boolean",
                        **params,
                    ),
                    row(
                        config,
                        index,
                        "optimizer_feasible",
                        float(result.feasible),
                        "boolean",
                        **params,
                    ),
                    row(config, index, "optimizer_accepted", float(accepted), "boolean", **params),
                    row(
                        config,
                        index,
                        "optimizer_iterations",
                        float(result.iterations),
                        "iterations",
                        **params,
                    ),
                    row(
                        config,
                        index,
                        "grid_best_capacity",
                        grid_best,
                        "bits_per_symbol",
                        estimator="binary_grid_validation",
                        grid_resolution=1001,
                        **params,
                    ),
                    row(
                        config,
                        index,
                        "optimizer_grid_difference",
                        abs(result.capacity_bits - grid_best),
                        "bits_per_symbol",
                        estimator="binary_grid_validation",
                        grid_resolution=1001,
                        **params,
                    ),
                ]
            )
            for symbol, probability in enumerate(result.input_probabilities):
                rows.append(
                    row(
                        config,
                        index,
                        "optimal_input_probability",
                        float(probability),
                        "probability",
                        symbol_index=symbol,
                        **params,
                    )
                )
        if len(rows) == 0:
            raise ValueError("detectability frontier requires a max_kl_bits sweep")
        return ExperimentOutput(rows=rows)
