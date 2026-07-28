"""Finite-sample likelihood-ratio detection campaign."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..channels import build_memoryless_channel
from ..information import blahut_arimoto
from ..information.divergence import kl_divergence
from ..quantization import UniformQuantizer
from .base import ExperimentContext, ExperimentOutput, ExperimentPlan
from .constrained import baseline_output
from .memoryless import make_jitter, row


def _scores(
    p1: np.ndarray, p0: np.ndarray, n: int, trials: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    log_ratio = np.full(len(p0), np.nan)
    both = (p1 > 0) & (p0 > 0)
    log_ratio[both] = np.log(p1[both] / p0[both])
    log_ratio[(p1 > 0) & (p0 == 0)] = np.inf
    log_ratio[(p1 == 0) & (p0 > 0)] = -np.inf
    if np.any(np.isnan(log_ratio)):
        log_ratio[np.isnan(log_ratio)] = 0.0
    baseline = rng.choice(len(p0), size=(trials, n), p=p0)
    active = rng.choice(len(p1), size=(trials, n), p=p1)
    return log_ratio[baseline].sum(axis=1), log_ratio[active].sum(axis=1)


def _roc(scores0: np.ndarray, scores1: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    thresholds = np.r_[np.inf, np.unique(np.r_[scores0, scores1])[::-1], -np.inf]
    fpr = np.array([(scores0 >= t).mean() for t in thresholds])
    tpr = np.array([(scores1 >= t).mean() for t in thresholds])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


class DetectionRunner:
    """Create scalar detector summaries and a ROC source table per sample size."""

    def plan(self, config, output_directory):
        return ExperimentPlan(
            config.experiment.kind,
            len(config.detection.sample_sizes),
            frozenset({"auc", "minimum_equal_prior_error"}),
            frozenset({"tables"}),
            output_directory,
        )

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
        if config.detection.active_distribution_source == "unconstrained_capacity":
            active = (
                blahut_arimoto(channel.probabilities).input_probabilities @ channel.probabilities
            )
        else:
            active = (
                np.asarray(
                    config.channel.input_probabilities
                    or np.full(len(channel.inputs), 1 / len(channel.inputs))
                )
                @ channel.probabilities
            )
        if np.allclose(active, baseline):
            raise ValueError(
                "finite-sample detection requires a distinct active and baseline distribution"
            )
        rows = []
        tables = context.directory / "tables"
        tables.mkdir(exist_ok=True)
        root = np.random.default_rng(context.root_seed_sequence)
        for index, n in enumerate(config.detection.sample_sizes):
            scores0, scores1 = _scores(
                active,
                baseline,
                n,
                config.detection.trials,
                np.random.default_rng(root.integers(2**32)),
            )
            fpr, tpr, auc = _roc(scores0, scores1)
            error = float(np.min((fpr + (1 - tpr)) / 2))
            roc_path = tables / f"roc_n_{n}.csv"
            pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr}).to_csv(
                roc_path, index=False
            )
            params: dict[str, object] = {
                "sample_size": n,
                "roc_table": str(roc_path.relative_to(context.directory)),
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
                        **params,
                    ),
                    row(
                        config,
                        index,
                        "minimum_equal_prior_error",
                        error,
                        "probability",
                        estimator="likelihood_ratio",
                        **params,
                    ),
                    row(
                        config,
                        index,
                        "theoretical_total_kl_bits",
                        n * kl_divergence(active, baseline),
                        "bits",
                        **params,
                    ),
                ]
            )
            for target in config.detection.target_false_positive_rates:
                admissible = tpr[fpr <= target]
                rows.append(
                    row(
                        config,
                        index,
                        "tpr_at_fpr",
                        float(admissible.max() if len(admissible) else 0),
                        "probability",
                        target_fpr=target,
                        **params,
                    )
                )
        return ExperimentOutput(rows=rows, artifacts=[Path("tables")])
