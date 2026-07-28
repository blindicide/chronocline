"""Experiment-specific plotting consumes only stored tidy result tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from chronocline.plotting import plot_result_directory


def rows_for(kind: str) -> list[dict[str, object]]:
    """Build minimal valid scientific axes for every plotting dispatch branch."""
    base = {"job_id": "a", "metric_value": 0.4}
    if kind in {"smoke", "memoryless_baseline"}:
        return [
            {**base, "metric_name": "mutual_information"},
            {**base, "metric_name": "capacity_bits_per_symbol", "metric_value": 0.6},
        ]
    if kind == "capacity_surface":
        return [
            {
                **base,
                "metric_name": "capacity_bits_per_symbol",
                "quantizer_step": step,
                "alphabet": alphabet,
            }
            for step, alphabet in [(0.25, "[0, 0.5]"), (0.5, "[0, 1]")]
        ]
    if kind == "phase_sensitivity":
        return [
            {**base, "metric_name": "capacity_bits_per_symbol", "quantizer_phase": phase}
            for phase in (0.0, 0.25)
        ]
    if kind == "detectability_frontier":
        return [
            {"job_id": job, "metric_name": metric, "metric_value": value}
            for job, value in [("a", 0.1), ("b", 0.2)]
            for metric in ["constrained_capacity_bits_per_symbol", "achieved_kl_bits"]
        ]
    if kind == "finite_sample_detection":
        return [
            {**base, "metric_name": "auc", "sample_size": size, "metric_value": value}
            for size, value in [(10, 0.6), (20, 0.8)]
        ]
    if kind == "alphabet_optimization":
        return [
            {
                **base,
                "metric_name": "optimized_alphabet",
                "symbol_index": index,
                "metric_value": value,
            }
            for index, value in enumerate([0.0, 1.0])
        ]
    if kind == "batching_comparison":
        return [
            {
                **base,
                "metric_name": "symbol_mutual_information",
                "batching_mode": mode,
                "batching_window": window,
                "metric_value": value,
            }
            for mode, window, value in [
                ("fixed_window_observation", 0.5, 0.2),
                ("ceiling_release", 1.0, 0.1),
            ]
        ]
    if kind == "jitter_comparison":
        return [
            {
                **base,
                "metric_name": "capacity_bits_per_symbol",
                "jitter_distribution": family,
                "quantizer_step": step,
                "metric_value": value,
            }
            for family, step, value in [("gaussian", 0.25, 0.6), ("laplace", 0.5, 0.4)]
        ]
    return [
        {**base, "metric_name": "capacity_bits_per_symbol", "quantizer_step": step}
        for step in (0.25, 0.5)
    ]


@pytest.mark.parametrize(
    "kind",
    [
        "smoke",
        "memoryless_baseline",
        "capacity_curve",
        "capacity_surface",
        "phase_sensitivity",
        "detectability_frontier",
        "finite_sample_detection",
        "alphabet_optimization",
        "jitter_comparison",
        "batching_comparison",
    ],
)
def test_every_plot_dispatch_writes_figure_and_source_table(kind: str, tmp_path: Path) -> None:
    """No plot relies on job/replication indices or invokes a scientific runner."""
    (tmp_path / "manifest.json").write_text(json.dumps({"experiment_kind": kind}))
    pd.DataFrame(rows_for(kind)).to_csv(tmp_path / "results.csv", index=False)

    image = plot_result_directory(tmp_path)

    assert image.exists()
    source = tmp_path / "tables" / f"figure_{image.stem}.csv"
    assert source.exists()
