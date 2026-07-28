"""Controlled stateful timestamp and batching comparison experiment."""

from __future__ import annotations

import numpy as np

from ..quantization import UniformQuantizer
from ..simulation import (
    block_mutual_information,
    cumulative_timestamp_observations,
    empirical_mutual_information,
)
from ..simulation.batching import ceiling_release, fixed_window
from .base import ExperimentContext, ExperimentOutput, ExperimentPlan
from .memoryless import make_jitter, row


class BatchingRunner:
    """Compare true delay observations with stateful timestamp/batching modes."""

    def plan(self, config, output_directory):
        return ExperimentPlan(
            config.experiment.kind,
            len(config.batching.modes) * len(config.batching.windows),
            frozenset(
                {
                    "symbol_mutual_information",
                    "zero_delay_probability",
                    "batch_size_mean",
                    "memoryless_approximation_error",
                    "plugin_block_mutual_information",
                }
            ),
            frozenset({"tables"}),
            output_directory,
        )

    def execute(self, context: ExperimentContext, jobs) -> ExperimentOutput:
        config = context.config
        rng = np.random.default_rng(context.root_seed_sequence)
        p = np.asarray(
            config.channel.input_probabilities
            or np.full(len(config.channel.alphabet.values), 1 / len(config.channel.alphabet.values))
        )
        symbols = rng.choice(len(p), config.simulation.trace_length, p=p)
        delays = np.asarray(config.channel.alphabet.values)[symbols]
        jitter = make_jitter(config).sample(len(delays) + 1, rng)
        timestamps, observed = cumulative_timestamp_observations(
            delays,
            jitter,
            UniformQuantizer(config.quantizer.step, config.quantizer.phase),
            preserve_order=config.simulation.preserve_order,
        )
        base_mi = empirical_mutual_information(
            symbols, np.rint(observed / config.quantizer.step).astype(int)
        )
        rows = []
        index = 0
        for mode in config.batching.modes:
            for window in config.batching.windows:
                if mode == "no_batching":
                    values = timestamps
                elif mode == "timestamp_quantization":
                    values = timestamps
                elif mode == "fixed_window_observation":
                    values = fixed_window(timestamps, window, config.batching.phase)
                else:
                    values = ceiling_release(timestamps, window, config.batching.phase)
                output = np.diff(values)
                mi = empirical_mutual_information(
                    symbols, np.rint(output / config.quantizer.step).astype(int)
                )
                params: dict[str, object] = {"batching_mode": mode, "batching_window": window}
                rows.extend(
                    [
                        row(
                            config,
                            index,
                            "symbol_mutual_information",
                            mi,
                            "bits_per_symbol",
                            estimator="empirical_stateful",
                            **params,
                        ),
                        row(
                            config,
                            index,
                            "zero_delay_probability",
                            float(np.mean(output == 0)),
                            "probability",
                            **params,
                        ),
                        row(
                            config,
                            index,
                            "batch_size_mean",
                            float(len(output) / max(1, len(np.unique(values)))),
                            "packets",
                            **params,
                        ),
                        row(
                            config,
                            index,
                            "memoryless_approximation_error",
                            abs(mi - base_mi),
                            "bits_per_symbol",
                            **params,
                        ),
                    ]
                )
                for block in config.simulation.block_lengths:
                    estimate = block_mutual_information(
                        symbols, np.rint(output / config.quantizer.step).astype(int), block
                    )
                    rows.append(
                        row(
                            config,
                            index,
                            "plugin_block_mutual_information",
                            float(estimate["block_mutual_information_estimate"]),
                            "bits_per_block",
                            block_length=block,
                            estimator="plugin_block",
                        )
                    )
                    rows.append(
                        row(
                            config,
                            index,
                            "normalized_plugin_block_mutual_information",
                            float(estimate["normalized_block_estimate"]),
                            "bits_per_symbol",
                            block_length=block,
                            estimator="plugin_block",
                        )
                    )
                index += 1
        return ExperimentOutput(rows=rows)
