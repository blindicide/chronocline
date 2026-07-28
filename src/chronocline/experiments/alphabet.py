"""Feasible constrained timing-alphabet optimisation campaign."""

from __future__ import annotations

import numpy as np

from ..channels import build_memoryless_channel
from ..information import blahut_arimoto
from ..information.constrained import constrained_capacity
from ..information.divergence import kl_divergence
from ..optimization import RestartCandidate, best_found_alphabet, binary_grid_optimize
from ..quantization import UniformQuantizer
from .base import ExperimentContext, ExperimentOutput, ExperimentPlan
from .constrained import baseline_output
from .memoryless import make_jitter, row


class AlphabetRunner:
    """Optimize only candidates that satisfy the configured scientific constraints."""

    def plan(self, config, output_directory):
        restarts = (
            config.alphabet_search.global_restarts if config.alphabet_search.symbols > 2 else 1
        )
        return ExperimentPlan(
            config.experiment.kind,
            restarts,
            frozenset({"optimized_alphabet", "best_found_capacity", "optimization_label"}),
            frozenset({"tables"}),
            output_directory,
        )

    @staticmethod
    def _evaluate(config, alphabet: np.ndarray) -> tuple[float, np.ndarray, float, bool, bool]:
        matrix = build_memoryless_channel(
            alphabet,
            make_jitter(config),
            UniformQuantizer(config.quantizer.step, config.quantizer.phase),
            tail_probability=config.matrix.tail_probability,
            include_overflow_bins=config.matrix.tail_mode == "overflow_bins",
        )
        baseline = baseline_output(config, matrix.probabilities)
        if (
            config.constraints.max_kl_bits is not None
            or config.constraints.max_mean_delay is not None
        ):
            result = constrained_capacity(
                matrix.probabilities,
                matrix.inputs,
                baseline,
                max_kl_bits=config.constraints.max_kl_bits,
                max_mean_delay=config.constraints.max_mean_delay,
                tolerance=config.optimization.tolerance,
                max_iterations=config.optimization.max_iterations,
                seed=config.experiment.seed,
            )
            return (
                result.capacity_bits,
                result.input_probabilities,
                result.kl_divergence_bits,
                result.feasible,
                result.converged,
            )
        unconstrained_result = blahut_arimoto(
            matrix.probabilities,
            tolerance=config.optimization.tolerance,
            max_iterations=config.optimization.max_iterations,
        )
        output = unconstrained_result.input_probabilities @ matrix.probabilities
        return (
            unconstrained_result.capacity_bits,
            unconstrained_result.input_probabilities,
            kl_divergence(output, baseline),
            True,
            True,
        )

    def execute(self, context: ExperimentContext, jobs) -> ExperimentOutput:
        config = context.config
        search = config.alphabet_search

        def objective(candidate: np.ndarray) -> float:
            try:
                value, _, _, feasible, converged = self._evaluate(config, candidate)
            except (RuntimeError, ValueError):
                return -np.inf
            return value if feasible and converged else -np.inf

        if search.symbols == 2:
            found = binary_grid_optimize(
                search.minimum,
                search.maximum,
                search.binary_grid_points,
                objective,
                min_spacing=search.minimum_spacing,
                anchor_first_symbol=search.anchor_first_symbol,
            )
            restart_records: tuple[RestartCandidate, ...] = ()
        else:
            found = best_found_alphabet(
                search.symbols,
                search.minimum,
                search.maximum,
                search.minimum_spacing,
                objective,
                seed=config.experiment.seed,
                anchor_first_symbol=search.anchor_first_symbol,
                global_restarts=search.global_restarts,
            )
            restart_records = found.restarts
        value, probabilities, achieved_kl, feasible, converged = self._evaluate(
            config, found.alphabet
        )
        if not feasible or not converged or not np.isfinite(value):
            raise RuntimeError("selected alphabet is not a feasible converged optimization")
        params: dict[str, object] = {
            "optimization_label": found.label,
            "anchor_first_symbol": search.anchor_first_symbol,
            "minimum_spacing": search.minimum_spacing,
            "optimizer_feasible": feasible,
            "optimizer_converged": converged,
        }
        rows = [
            row(
                config,
                0,
                "optimized_alphabet",
                float(item),
                "normalized_time",
                symbol_index=index,
                **params,
            )
            for index, item in enumerate(found.alphabet)
        ]
        rows.extend(
            [
                row(config, 0, "best_found_capacity", value, "bits_per_symbol", **params),
                row(config, 0, "achieved_kl_bits", achieved_kl, "bits", **params),
                row(config, 0, "optimizer_converged", float(converged), "boolean", **params),
                row(config, 0, "optimizer_feasible", float(feasible), "boolean", **params),
                row(config, 0, "optimizer_accepted", 1.0, "boolean", **params),
                row(config, 0, "optimization_label", 1.0, "code", label=found.label, **params),
            ]
        )
        rows.extend(
            row(
                config,
                0,
                "optimal_input_probability",
                float(item),
                "probability",
                symbol_index=index,
                **params,
            )
            for index, item in enumerate(probabilities)
        )
        for restart_index, restart in enumerate(restart_records):
            restart_params = {
                **params,
                "restart_index": restart_index,
                "restart_seed": restart.seed,
                "restart_alphabet": str(restart.alphabet.tolist()),
            }
            rows.extend(
                [
                    row(
                        config,
                        restart_index,
                        "optimizer_accepted",
                        float(restart.accepted),
                        "boolean",
                        **restart_params,
                    ),
                    row(
                        config,
                        restart_index,
                        "best_found_capacity",
                        restart.objective,
                        "bits_per_symbol",
                        estimator="restart_candidate",
                        **restart_params,
                    ),
                ]
            )
        return ExperimentOutput(rows=rows)
