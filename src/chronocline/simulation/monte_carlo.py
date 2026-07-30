"""Empirical information estimates for stateful traces."""

from __future__ import annotations

import numpy as np


def empirical_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Plug-in I(X;Y) for discrete samples; not channel capacity with memory."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) != len(y):
        raise ValueError("sample vectors must have equal length")
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((xi.max() + 1, yi.max() + 1))
    np.add.at(joint, (xi, yi), 1)
    joint /= len(x)
    px, py = joint.sum(1), joint.sum(0)
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / (px[:, None] * py)[mask])))


def block_mutual_information(x: np.ndarray, y: np.ndarray, block: int) -> dict[str, float]:
    """Return row-wise encoded block MI with Miller--Madow diagnostics."""
    if block < 1:
        raise ValueError("block must be positive")
    n = len(x) // block
    xb = np.asarray(x)[: n * block].reshape(n, block)
    yb = np.asarray(y)[: n * block].reshape(n, block)
    _, x_codes = np.unique(xb, axis=0, return_inverse=True)
    _, y_codes = np.unique(yb, axis=0, return_inverse=True)
    value = empirical_mutual_information(x_codes, y_codes)
    input_states = len(np.unique(x_codes))
    output_states = len(np.unique(y_codes))
    joint_states = len(np.unique(np.column_stack((x_codes, y_codes)), axis=0))
    # H_MM = H_plugin + (K - 1)/(2N ln 2); MI combines three entropy terms.
    correction = (input_states + output_states - joint_states - 1) / (2 * n * np.log(2))
    miller_madow = value + correction
    return {
        "block_length": block,
        "block_mutual_information_estimate": value,
        "normalized_block_estimate": value / block,
        "miller_madow_block_mutual_information": miller_madow,
        "normalized_miller_madow_block_estimate": miller_madow / block,
        "miller_madow_was_clipped": 0.0,
        "observed_input_states": float(input_states),
        "observed_output_states": float(output_states),
        "observed_joint_states": float(joint_states),
        "available_blocks": float(n),
        "samples_per_joint_state": float(n / joint_states),
        "undersampling_warning": float(n < 10 * joint_states),
    }
