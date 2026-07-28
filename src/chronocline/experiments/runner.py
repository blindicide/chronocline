"""Deterministic, resumable experiment execution."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from ..channels import build_memoryless_channel, monte_carlo_matrix
from ..config import RunConfig
from ..distributions import GaussianMixture, gaussian, laplace, student_t, uniform
from ..distributions.base import JitterDistribution
from ..information import blahut_arimoto, mutual_information
from ..information.divergence import kl_divergence
from ..quantization import UniformQuantizer
from ..results.manifest import create_manifest, finalize_manifest
from ..results.storage import write_results


def make_jitter(config: RunConfig) -> JitterDistribution:
    """Build a configured independent jitter distribution."""
    j = config.jitter
    if j.distribution == "gaussian":
        return gaussian(j.mean, j.scale)
    if j.distribution == "laplace":
        return laplace(j.mean, j.scale)
    if j.distribution == "uniform":
        assert j.lower is not None and j.upper is not None
        return uniform(j.lower, j.upper)
    if j.distribution == "student_t":
        assert j.degrees_of_freedom is not None
        return student_t(j.degrees_of_freedom, j.mean, j.scale)
    return GaussianMixture(j.weights or [], j.means or [], j.scales or [])


def sweep_combinations(config: RunConfig) -> list[dict[str, object]]:
    """Resolve cartesian sweep parameters deterministically."""
    keys = sorted(config.sweep.parameters)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(config.sweep.parameters[k] for k in keys))
    ] or [{}]


def run(config: RunConfig, *, dry_run: bool = False) -> Path | dict[str, object]:
    """Run memoryless sweeps, saving an atomic traceable result bundle."""
    combinations = sweep_combinations(config)
    base = Path(config.experiment.output_directory) / config.experiment.name
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    run_id = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    directory = base / run_id
    if dry_run:
        return {
            "jobs": len(combinations),
            "expected_rows": len(combinations) * 4,
            "output_directory": str(directory),
            "workers": config.experiment.workers,
        }
    if (
        directory.exists()
        and (directory / "manifest.json").exists()
        and not config.experiment.overwrite
    ):
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.original.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )
    (directory / "config.resolved.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True)
    )
    manifest = create_manifest(
        hashlib.sha256(canonical.encode()).hexdigest(),
        config.experiment.name,
        config.experiment.seed,
        config.experiment.workers,
        config.experiment.locale,
    )
    rows: list[dict[str, object]] = []
    for stream, overrides in enumerate(combinations):
        step_value = overrides.get("quantizer.step", config.quantizer.step)
        if not isinstance(step_value, (int, float)):
            raise ValueError("quantizer.step sweep values must be numeric")
        step = float(step_value)
        q = UniformQuantizer(step, config.quantizer.phase, config.quantizer.mode)
        channel = build_memoryless_channel(
            config.channel.alphabet.values,
            make_jitter(config),
            q,
            tail_probability=config.matrix.tail_probability,
            include_overflow_bins=config.matrix.include_overflow_bins,
        )
        p = np.asarray(
            config.channel.input_probabilities
            or np.full(len(channel.inputs), 1 / len(channel.inputs))
        )
        capacity = blahut_arimoto(
            channel.probabilities,
            tolerance=config.optimization.tolerance,
            max_iterations=config.optimization.max_iterations,
        )
        output = p @ channel.probabilities
        shared = {
            "experiment": config.experiment.name,
            "replication": stream,
            "units": "bits_per_symbol",
            "estimator": "exact_cdf_difference",
            "status": "complete",
            **overrides,
        }
        rows.extend(
            [
                {
                    **shared,
                    "metric_name": "mutual_information",
                    "metric_value": mutual_information(p, channel.probabilities),
                },
                {
                    **shared,
                    "metric_name": "capacity_bits_per_symbol",
                    "metric_value": capacity.capacity_bits,
                },
                {**shared, "metric_name": "capacity_residual", "metric_value": capacity.residual},
                {
                    **shared,
                    "metric_name": "matrix_row_sum_error",
                    "metric_value": channel.row_sum_error,
                },
            ]
        )
        np.savez_compressed(
            directory / f"matrix_{stream}.npz",
            probabilities=channel.probabilities,
            inputs=channel.inputs,
            outputs=channel.outputs,
        )
        if config.experiment.name in {"smoke", "memoryless_baseline"}:
            empirical = monte_carlo_matrix(
                channel,
                make_jitter(config),
                q,
                20_000,
                np.random.default_rng(
                    np.random.SeedSequence(config.experiment.seed, spawn_key=(stream,))
                ),
            )
            rows.append(
                {
                    **shared,
                    "metric_name": "monte_carlo_max_absolute_error",
                    "metric_value": float(np.max(np.abs(empirical - channel.probabilities))),
                }
            )
        if config.constraints.max_kl_divergence is not None:
            rows.append(
                {
                    **shared,
                    "metric_name": "active_output_kl_bits",
                    "metric_value": kl_divergence(output, output),
                }
            )
    write_results(directory, rows)
    finalize_manifest(directory, manifest)
    return directory
