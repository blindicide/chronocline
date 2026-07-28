"""Finite-sample likelihood-ratio detection campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..channels import build_memoryless_channel
from ..information import blahut_arimoto
from ..information.constrained import constrained_capacity
from ..information.divergence import js_divergence, kl_divergence, total_variation
from ..quantization import UniformQuantizer
from .base import ExperimentContext, ExperimentOutput, ExperimentPlan
from .constrained import baseline_output
from .memoryless import make_jitter, row


def _scores(
    p1: np.ndarray,
    p0: np.ndarray,
    n: int,
    trials: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw deterministic score batches without a maximum-size trial array."""
    log_ratio = np.full(len(p0), np.nan)
    both = (p1 > 0) & (p0 > 0)
    log_ratio[both] = np.log(p1[both] / p0[both])
    log_ratio[(p1 > 0) & (p0 == 0)] = np.inf
    log_ratio[(p1 == 0) & (p0 > 0)] = -np.inf
    if np.any(np.isnan(log_ratio)):
        log_ratio[np.isnan(log_ratio)] = 0.0
    baseline_scores: list[np.ndarray] = []
    active_scores: list[np.ndarray] = []
    for start in range(0, trials, batch_size):
        count = min(batch_size, trials - start)
        baseline = rng.choice(len(p0), size=(count, n), p=p0)
        active = rng.choice(len(p1), size=(count, n), p=p1)
        baseline_scores.append(log_ratio[baseline].sum(axis=1))
        active_scores.append(log_ratio[active].sum(axis=1))
    return np.concatenate(baseline_scores), np.concatenate(active_scores)


def _roc(scores0: np.ndarray, scores1: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC endpoints by sorted-score ranks rather than quadratic scans."""
    thresholds = np.r_[np.inf, np.unique(np.r_[scores0, scores1])[::-1], -np.inf]
    fpr = np.asarray(
        (len(scores0) - np.searchsorted(np.sort(scores0), thresholds, side="left")) / len(scores0)
    )
    tpr = np.asarray(
        (len(scores1) - np.searchsorted(np.sort(scores1), thresholds, side="left")) / len(scores1)
    )
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def _bootstrap_metrics(
    scores0: np.ndarray,
    scores1: np.ndarray,
    repetitions: int,
    target_fprs: list[float],
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    """Return percentile intervals for detector summaries using independent resamples."""
    samples: dict[str, list[float]] = {"auc": [], "minimum_equal_prior_error": []}
    samples.update({f"tpr_at_fpr_{rate:g}": [] for rate in target_fprs})
    for _ in range(repetitions):
        first = rng.choice(scores0, len(scores0), replace=True)
        second = rng.choice(scores1, len(scores1), replace=True)
        fpr, tpr, auc = _roc(first, second)
        samples["auc"].append(auc)
        samples["minimum_equal_prior_error"].append(float(np.min((fpr + (1 - tpr)) / 2)))
        for rate in target_fprs:
            samples[f"tpr_at_fpr_{rate:g}"].append(float(np.max(tpr[fpr <= rate])))
    return {
        name: tuple(np.quantile(values, [0.025, 0.975]))  # type: ignore[return-value]
        for name, values in samples.items()
    }


class DetectionRunner:
    """Create scalar detector summaries and a ROC source table per sample size."""

    def plan(self, config, output_directory):
        return ExperimentPlan(
            config.experiment.kind,
            len(config.detection.sample_sizes) * 2,
            frozenset({"auc", "minimum_equal_prior_error"}),
            frozenset({"tables"}),
            output_directory,
        )

    @staticmethod
    def _active_output(config, channel, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resolve the configured active distribution without implicit fallthrough."""
        source = config.detection.active_distribution_source
        if source == "configured_input":
            probabilities = np.asarray(
                config.channel.input_probabilities
                or np.full(len(channel.inputs), 1 / len(channel.inputs))
            )
        elif source == "unconstrained_capacity":
            probabilities = blahut_arimoto(
                channel.probabilities,
                tolerance=config.optimization.tolerance,
                max_iterations=config.optimization.max_iterations,
            ).input_probabilities
        elif source == "constrained_optimum":
            if (
                config.constraints.max_kl_bits is None
                and config.constraints.max_mean_delay is None
            ):
                raise ValueError("constrained_optimum requires at least one active constraint")
            result = constrained_capacity(
                channel.probabilities,
                channel.inputs,
                baseline,
                max_kl_bits=config.constraints.max_kl_bits,
                max_mean_delay=config.constraints.max_mean_delay,
                tolerance=config.optimization.tolerance,
                max_iterations=config.optimization.max_iterations,
                seed=config.experiment.seed,
            )
            if not result.converged or not result.feasible:
                raise RuntimeError(
                    "constrained active-distribution optimization did not converge feasibly"
                )
            probabilities = result.input_probabilities
        else:  # Defensive protection for direct construction that bypasses Pydantic.
            raise ValueError(f"unsupported active distribution source: {source}")
        return probabilities @ channel.probabilities, probabilities

    def execute(self, context: ExperimentContext, jobs) -> ExperimentOutput:
        config = context.config
        q = UniformQuantizer(config.quantizer.step, config.quantizer.phase, config.quantizer.mode)
        channel = build_memoryless_channel(
            config.channel.alphabet.values,
            make_jitter(config),
            q,
            tail_probability=config.matrix.tail_probability,
            include_overflow_bins=True,
        )
        baseline = baseline_output(config, channel.probabilities)
        active, active_input = self._active_output(config, channel, baseline)
        if np.allclose(active, baseline):
            raise ValueError(
                "finite-sample detection requires a distinct active and baseline distribution"
            )
        rows = []
        tables = context.directory / "tables"
        tables.mkdir(exist_ok=True)
        root = np.random.default_rng(context.root_seed_sequence)
        pairs = {
            "active_vs_baseline": (active, baseline),
            "baseline_vs_baseline": (baseline, baseline),
        }
        for size_index, n in enumerate(config.detection.sample_sizes):
            for pair_index, (pair_name, (alternative, null)) in enumerate(pairs.items()):
                index = size_index * len(pairs) + pair_index
                scores0, scores1 = _scores(
                    alternative,
                    null,
                    n,
                    config.detection.trials,
                    config.detection.batch_size,
                    np.random.default_rng(root.integers(2**32)),
                )
                fpr, tpr, auc = _roc(scores0, scores1)
                error = float(np.min((fpr + (1 - tpr)) / 2))
                intervals = _bootstrap_metrics(
                    scores0,
                    scores1,
                    config.detection.bootstrap_repetitions,
                    config.detection.target_false_positive_rates,
                    np.random.default_rng(root.integers(2**32)),
                )
                roc_path = tables / f"roc_n_{n}_{pair_name}.csv"
                pd.DataFrame(
                    {
                        "false_positive_rate": fpr,
                        "true_positive_rate": tpr,
                        "hypothesis_pair": pair_name,
                    }
                ).to_csv(roc_path, index=False)
                params: dict[str, object] = {
                    "sample_size": n,
                    "roc_table": str(roc_path.relative_to(context.directory)),
                    "hypothesis_pair": pair_name,
                    "active_distribution_source": config.detection.active_distribution_source,
                    "active_input_probabilities": json.dumps(active_input.tolist()),
                    "active_output_hash": hashlib.sha256(
                        np.asarray(alternative, dtype=np.float64).tobytes()
                    ).hexdigest(),
                    "trial_batch_size": config.detection.batch_size,
                    "total_trials": config.detection.trials,
                    "root_seed": config.experiment.seed,
                    "replication": 0,
                }
                rows.extend(
                    [
                        row(
                            config,
                            index,
                            "auc",
                            auc,
                            "dimensionless",
                            estimator="likelihood_ratio",
                            confidence_interval_lower=intervals["auc"][0],
                            confidence_interval_upper=intervals["auc"][1],
                            **params,
                        ),
                        row(
                            config,
                            index,
                            "minimum_equal_prior_error",
                            error,
                            "probability",
                            estimator="likelihood_ratio",
                            confidence_interval_lower=intervals["minimum_equal_prior_error"][0],
                            confidence_interval_upper=intervals["minimum_equal_prior_error"][1],
                            **params,
                        ),
                        row(
                            config,
                            index,
                            "theoretical_total_kl_bits",
                            n * kl_divergence(alternative, null),
                            "bits",
                            **params,
                        ),
                        row(
                            config,
                            index,
                            "theoretical_per_observation_kl_bits",
                            kl_divergence(alternative, null),
                            "bits",
                            **params,
                        ),
                        row(
                            config,
                            index,
                            "jensen_shannon_divergence_bits",
                            js_divergence(alternative, null),
                            "bits",
                            **params,
                        ),
                        row(
                            config,
                            index,
                            "total_variation_distance",
                            total_variation(alternative, null),
                            "probability",
                            **params,
                        ),
                    ]
                )
                for target in config.detection.target_false_positive_rates:
                    name = f"tpr_at_fpr_{target:g}"
                    rows.append(
                        row(
                            config,
                            index,
                            "tpr_at_fpr",
                            float(np.max(tpr[fpr <= target])),
                            "probability",
                            estimator="likelihood_ratio",
                            target_false_positive_rate=target,
                            confidence_interval_lower=intervals[name][0],
                            confidence_interval_upper=intervals[name][1],
                            **params,
                        )
                    )
        return ExperimentOutput(rows=rows, artifacts=[Path("tables")])
