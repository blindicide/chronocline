"""Replicated stateful observation and batching comparison experiment."""

from __future__ import annotations

import numpy as np

from ..quantization import UniformQuantizer
from ..simulation import (
    batch_observation_trace,
    block_mutual_information,
    empirical_mutual_information,
    observation_trace,
)
from .base import ExperimentContext, ExperimentOutput, ExperimentPlan
from .memoryless import make_jitter, row


class BatchingRunner:
    """Compare distinct X→S→A→R→Y models with actual configured replications."""

    def plan(self, config, output_directory):
        return ExperimentPlan(
            config.experiment.kind,
            len(config.batching.modes)
            * len(config.batching.windows)
            * config.simulation.replications,
            frozenset(
                {
                    "symbol_mutual_information",
                    "zero_delay_probability",
                    "batch_size_mean",
                    "batch_size_maximum",
                    "memoryless_approximation_error",
                    "plugin_block_mutual_information",
                }
            ),
            frozenset({"tables"}),
            output_directory,
        )

    @staticmethod
    def _trace(config, delays: np.ndarray, jitter: np.ndarray, mode: str):
        base_mode = mode
        if mode == "no_batching":
            base_mode = "ideal_delays"
        elif mode in {"fixed_window_observation", "ceiling_release"}:
            base_mode = "arrival_timestamps"
        if mode == "timestamp_quantization_then_batching":
            base_mode = "timestamp_quantization_then_batching"
        return observation_trace(
            delays,
            jitter,
            UniformQuantizer(config.quantizer.step, config.quantizer.phase),
            model=base_mode,
            preserve_order=config.simulation.preserve_order,
        )

    def execute(self, context: ExperimentContext, jobs) -> ExperimentOutput:
        config = context.config
        root = np.random.default_rng(context.root_seed_sequence)
        p = np.asarray(
            config.channel.input_probabilities
            or np.full(len(config.channel.alphabet.values), 1 / len(config.channel.alphabet.values))
        )
        alphabet = np.asarray(config.channel.alphabet.values)
        rows: list[dict[str, object]] = []
        job_index = 0
        transient = config.simulation.transient_observations
        for replication in range(config.simulation.replications):
            rng = np.random.default_rng(root.integers(2**32))
            symbols = rng.choice(len(p), config.simulation.trace_length, p=p)
            delays = alphabet[symbols]
            jitter = make_jitter(config).sample(len(delays), rng)
            for mode in config.batching.modes:
                for window in config.batching.windows:
                    trace = self._trace(config, delays, jitter, mode)
                    if mode in {
                        "fixed_window_observation",
                        "ceiling_release",
                        "timestamp_quantization_then_batching",
                    }:
                        trace = batch_observation_trace(
                            trace,
                            window,
                            config.batching.phase,
                            ceiling=mode == "ceiling_release",
                            maximum_batch_size=config.batching.maximum_batch_size,
                        )
                    start = min(transient, len(symbols) - 1)
                    observed = trace.observed_delays[start:]
                    aligned_symbols = symbols[start:]
                    encoded = np.rint(observed / config.quantizer.step).astype(int)
                    mi = empirical_mutual_information(aligned_symbols, encoded)
                    ideal = empirical_mutual_information(symbols[start:], symbols[start:])
                    _, batch_sizes = np.unique(trace.batch_ids[start:], return_counts=True)
                    params: dict[str, object] = {
                        "observation_model": trace.model,
                        "batching_mode": mode,
                        "batching_window": window,
                        "batching_phase": config.batching.phase,
                        "maximum_batch_size": config.batching.maximum_batch_size,
                        "replication": replication,
                        "trace_length": config.simulation.trace_length,
                        "transient_observations": transient,
                        "jitter_application": config.simulation.jitter_application,
                        "preserve_order": config.simulation.preserve_order,
                    }
                    rows.extend(
                        [
                            row(
                                config,
                                job_index,
                                "symbol_mutual_information",
                                mi,
                                "bits_per_symbol",
                                estimator="empirical_stateful",
                                **params,
                            ),
                            row(
                                config,
                                job_index,
                                "zero_delay_probability",
                                float(np.mean(observed == 0)),
                                "probability",
                                **params,
                            ),
                            row(
                                config,
                                job_index,
                                "batch_size_mean",
                                float(batch_sizes.mean()),
                                "packets",
                                **params,
                            ),
                            row(
                                config,
                                job_index,
                                "batch_size_maximum",
                                float(batch_sizes.max()),
                                "packets",
                                **params,
                            ),
                            row(
                                config,
                                job_index,
                                "memoryless_approximation_error",
                                abs(mi - ideal),
                                "bits_per_symbol",
                                **params,
                            ),
                        ]
                    )
                    for block in config.simulation.block_lengths:
                        estimate = block_mutual_information(aligned_symbols, encoded, block)
                        block_params = {
                            "observed_input_states": estimate["observed_input_states"],
                            "observed_output_states": estimate["observed_output_states"],
                            "observed_joint_states": estimate["observed_joint_states"],
                            "available_blocks": estimate["available_blocks"],
                            "samples_per_joint_state": estimate["samples_per_joint_state"],
                            "undersampling_warning": estimate["undersampling_warning"],
                        }
                        for metric, unit in (
                            ("plugin_block_mutual_information", "bits_per_block"),
                            ("normalized_plugin_block_mutual_information", "bits_per_symbol"),
                            ("miller_madow_block_mutual_information", "bits_per_block"),
                            (
                                "normalized_miller_madow_block_mutual_information",
                                "bits_per_symbol",
                            ),
                        ):
                            if metric == "plugin_block_mutual_information":
                                key = "block_mutual_information_estimate"
                            elif metric == "normalized_plugin_block_mutual_information":
                                key = "normalized_block_estimate"
                            elif metric == "miller_madow_block_mutual_information":
                                key = "miller_madow_block_mutual_information"
                            else:
                                key = "normalized_miller_madow_block_estimate"
                            rows.append(
                                row(
                                    config,
                                    job_index,
                                    metric,
                                    float(estimate[key]),
                                    unit,
                                    block_length=block,
                                    estimator="plugin_block",
                                    **params,
                                    **block_params,
                                )
                            )
                    job_index += 1
        return ExperimentOutput(rows=rows)
