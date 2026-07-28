"""Validated schema-2 scalar results and units."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from ..config import ExperimentKind

METRIC_UNITS = {
    "mutual_information": "bits_per_symbol",
    "capacity_bits_per_symbol": "bits_per_symbol",
    "normalized_capacity": "probability",
    "capacity_loss": "bits_per_symbol",
    "capacity_residual": "bits_per_symbol",
    "matrix_row_sum_error": "probability",
    "monte_carlo_max_absolute_error": "probability",
    "active_output_kl_bits": "bits",
    "achieved_kl_bits": "bits",
    "constrained_capacity_bits_per_symbol": "bits_per_symbol",
    "mean_delay": "normalized_time",
    "optimizer_converged": "boolean",
    "optimizer_feasible": "boolean",
    "optimizer_accepted": "boolean",
    "optimizer_iterations": "iterations",
    "optimizer_grid_difference": "bits_per_symbol",
    "grid_best_capacity": "bits_per_symbol",
    "hard_decision_error": "probability",
    "hard_decision_capacity": "bits_per_symbol",
    "hard_decision_information_loss": "bits_per_symbol",
    "quantizer_phase": "normalized_time",
    "optimal_input_probability": "probability",
    "auc": "dimensionless",
    "minimum_equal_prior_error": "probability",
    "tpr_at_fpr": "probability",
    "theoretical_total_kl_bits": "bits",
    "theoretical_per_observation_kl_bits": "bits",
    "jensen_shannon_divergence_bits": "bits",
    "total_variation_distance": "probability",
    "symbol_mutual_information": "bits_per_symbol",
    "zero_delay_probability": "probability",
    "batch_size_mean": "packets",
    "batch_size_maximum": "packets",
    "memoryless_approximation_error": "bits_per_symbol",
    "plugin_block_mutual_information": "bits_per_block",
    "normalized_plugin_block_mutual_information": "bits_per_symbol",
    "optimized_alphabet": "normalized_time",
    "best_found_capacity": "bits_per_symbol",
    "optimization_label": "code",
    "jitter_variance": "normalized_time_squared",
    "relative_capacity_difference_from_gaussian": "probability",
    "capacity_phase_minimum": "bits_per_symbol",
    "capacity_phase_maximum": "bits_per_symbol",
    "capacity_phase_mean": "bits_per_symbol",
}


class ResultRow(BaseModel):
    """One scalar scientific result with explicit provenance."""

    model_config = ConfigDict(extra="allow")
    experiment_name: str
    experiment_kind: ExperimentKind
    job_id: str
    sweep_index: int
    replication: int | None = None
    metric_name: str
    metric_value: float
    units: str
    estimator: str
    status: str
    warning_code: str | None = None
    confidence_interval_lower: float | None = None
    confidence_interval_upper: float | None = None

    @field_validator("units")
    @classmethod
    def nonempty_unit(cls, value: str) -> str:
        if not value:
            raise ValueError("units may not be empty")
        return value

    def validate_metric_unit(self) -> None:
        """Verify a metric's registered canonical unit."""
        expected = METRIC_UNITS.get(self.metric_name)
        if expected is None:
            raise ValueError(f"unregistered metric {self.metric_name}")
        if self.units != expected:
            raise ValueError(f"metric {self.metric_name} requires units {expected}")
