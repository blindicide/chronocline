"""Actual grid-based timing alphabet optimisation campaign."""

from __future__ import annotations

import numpy as np

from ..channels import build_memoryless_channel
from ..information import blahut_arimoto
from ..quantization import UniformQuantizer
from .base import ExperimentContext, ExperimentOutput, ExperimentPlan
from .memoryless import make_jitter, row


class AlphabetRunner:
    """Find a binary exact-on-grid timing alphabet rather than evaluating a seed alphabet."""

    def plan(self, config, output_directory):
        return ExperimentPlan(
            config.experiment.kind,
            config.alphabet_search.binary_grid_points,
            frozenset({"optimized_alphabet", "best_found_capacity", "optimization_label"}),
            frozenset({"tables"}),
            output_directory,
        )

    def execute(self, context: ExperimentContext, jobs) -> ExperimentOutput:
        config = context.config
        search = config.alphabet_search
        candidates = np.linspace(
            search.minimum + search.minimum_spacing, search.maximum, search.binary_grid_points
        )
        best: tuple[float, list[float] | None, np.ndarray | None] = (-np.inf, None, None)
        for distance in candidates:
            alphabet = (
                [search.minimum, float(distance)]
                if search.anchor_first_symbol
                else [float(distance - search.minimum_spacing), float(distance)]
            )
            if alphabet[1] - alphabet[0] < search.minimum_spacing:
                continue
            matrix = build_memoryless_channel(
                alphabet,
                make_jitter(config),
                UniformQuantizer(config.quantizer.step, config.quantizer.phase),
                tail_probability=config.matrix.tail_probability,
                include_overflow_bins=True,
            )
            result = blahut_arimoto(
                matrix.probabilities,
                tolerance=config.optimization.tolerance,
                max_iterations=config.optimization.max_iterations,
            )
            if result.capacity_bits > best[0]:
                best = (result.capacity_bits, alphabet, result.input_probabilities)
        value, candidate_alphabet, candidate_probabilities = best
        if candidate_alphabet is None or candidate_probabilities is None:
            raise ValueError("alphabet search found no admissible candidate")
        alphabet = candidate_alphabet
        probabilities = candidate_probabilities
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
        rows.append(row(config, 0, "optimization_label", 1.0, "code", label="exact_grid_optimum"))
        rows.extend(
            row(config, 0, "optimal_input_probability", float(p), "probability", symbol_index=i)
            for i, p in enumerate(probabilities)
        )
        return ExperimentOutput(rows=rows)
