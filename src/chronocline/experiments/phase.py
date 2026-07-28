"""Fixed-step quantizer phase sensitivity experiment."""

from __future__ import annotations

from typing import cast

import numpy as np

from ..config import RunConfig
from .base import ExperimentContext, ExperimentOutput, ExperimentPlan
from .memoryless import evaluate_memoryless, row


class PhaseRunner:
    """Evaluate capacity and MI for explicitly swept fixed phases."""

    def plan(self, config: RunConfig, output_directory):
        return ExperimentPlan(
            config.experiment.kind,
            len(config.sweep.parameters.get("quantizer.phase", [])),
            frozenset({"quantizer_phase", "capacity_bits_per_symbol"}),
            frozenset({"tables/figure_capacity_vs_phase.csv"}),
            output_directory,
        )

    def execute(
        self, context: ExperimentContext, jobs: list[tuple[int, RunConfig]]
    ) -> ExperimentOutput:
        rows = []
        for index, config in jobs:
            rows.extend(evaluate_memoryless(config, index, context.directory)[0])
            rows.append(
                row(config, index, "quantizer_phase", config.quantizer.phase, "normalized_time")
            )
        phase_values = [
            float(cast(float, r["metric_value"]))
            for r in rows
            if r["metric_name"] == "quantizer_phase"
        ]
        if len(set(phase_values)) < 2:
            raise ValueError("phase sensitivity requires multiple distinct phases")
        capacities = [
            float(cast(float, r["metric_value"]))
            for r in rows
            if r["metric_name"] == "capacity_bits_per_symbol"
        ]
        rows.extend(
            [
                row(
                    context.config, 0, "capacity_phase_minimum", min(capacities), "bits_per_symbol"
                ),
                row(
                    context.config, 0, "capacity_phase_maximum", max(capacities), "bits_per_symbol"
                ),
                row(
                    context.config,
                    0,
                    "capacity_phase_mean",
                    float(np.mean(capacities)),
                    "bits_per_symbol",
                ),
            ]
        )
        return ExperimentOutput(rows=rows)
