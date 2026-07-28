"""Actual grid-based timing alphabet optimisation campaign."""

from __future__ import annotations

import numpy as np

from ..channels import build_memoryless_channel
from ..information import blahut_arimoto
from ..information.constrained import constrained_capacity
from ..optimization import best_found_alphabet
from ..quantization import UniformQuantizer
from .base import ExperimentContext, ExperimentOutput, ExperimentPlan
from .memoryless import make_jitter, row


class AlphabetRunner:
    """Find a binary exact-on-grid timing alphabet rather than evaluating a seed alphabet."""

    def plan(self, config, output_directory):
        return ExperimentPlan(
            config.experiment.kind,
            1,
            frozenset({"optimized_alphabet", "best_found_capacity", "optimization_label"}),
            frozenset({"tables"}),
            output_directory,
        )

    def execute(self, context: ExperimentContext, jobs) -> ExperimentOutput:
        config = context.config
        search = config.alphabet_search
        def capacity(alphabet: np.ndarray) -> float:
            matrix = build_memoryless_channel(
                alphabet,
                make_jitter(config),
                UniformQuantizer(config.quantizer.step, config.quantizer.phase),
                tail_probability=config.matrix.tail_probability,
                include_overflow_bins=True,
            )
            return blahut_arimoto(
                matrix.probabilities,
                tolerance=config.optimization.tolerance,
                max_iterations=config.optimization.max_iterations,
            ).capacity_bits

        if search.symbols == 2:
            candidates = np.linspace(
                search.minimum + search.minimum_spacing,
                search.maximum,
                search.binary_grid_points,
            )
            evaluated = [
                np.array([search.minimum, distance])
                for distance in candidates
                if distance - search.minimum >= search.minimum_spacing
            ]
            alphabet = max(evaluated, key=capacity)
            label = "exact_grid_optimum"
        else:
            found = best_found_alphabet(
                search.symbols,
                search.minimum,
                search.maximum,
                search.minimum_spacing,
                capacity,
                seed=config.experiment.seed,
            )
            alphabet = found.alphabet
            label = found.label
        matrix = build_memoryless_channel(
            alphabet,
            make_jitter(config),
            UniformQuantizer(config.quantizer.step, config.quantizer.phase),
            tail_probability=config.matrix.tail_probability,
            include_overflow_bins=True,
        )
        baseline = matrix.probabilities[config.baseline.symbol_index]
        if (
            config.constraints.max_kl_bits is not None
            or config.constraints.max_mean_delay is not None
        ):
            constrained = constrained_capacity(
                matrix.probabilities,
                matrix.inputs,
                baseline,
                max_kl_bits=config.constraints.max_kl_bits,
                max_mean_delay=config.constraints.max_mean_delay,
                tolerance=config.optimization.tolerance,
                max_iterations=config.optimization.max_iterations,
                seed=config.experiment.seed,
            )
            value, probabilities = constrained.capacity_bits, constrained.input_probabilities
        else:
            unconstrained = blahut_arimoto(
                matrix.probabilities,
                tolerance=config.optimization.tolerance,
                max_iterations=config.optimization.max_iterations,
            )
            value, probabilities = unconstrained.capacity_bits, unconstrained.input_probabilities
        rows = [
            row(config, 0, "optimized_alphabet", float(v), "normalized_time", symbol_index=i)
            for i, v in enumerate(alphabet)
        ]
        rows.append(
            row(
                config,
                0,
                "best_found_capacity",
                float(value),
                "bits_per_symbol",
                estimator="grid_search",
            )
        )
        rows.append(row(config, 0, "optimization_label", 1.0, "code", label=label))
        rows.extend(
            row(config, 0, "optimal_input_probability", float(p), "probability", symbol_index=i)
            for i, p in enumerate(probabilities)
        )
        return ExperimentOutput(rows=rows)
