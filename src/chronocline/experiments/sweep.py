"""Generic cartesian dotted configuration sweep resolution."""

from __future__ import annotations

import itertools
from typing import Any

from ..config import RunConfig, apply_overrides


def resolve_sweep(
    config: RunConfig, allowed_paths: set[str]
) -> list[tuple[int, RunConfig, dict[str, Any]]]:
    """Resolve each override and reject a path unused by a specialised runner."""
    unknown = set(config.sweep.parameters) - allowed_paths
    if unknown:
        raise ValueError(f"unsupported sweep paths for {config.experiment.kind}: {sorted(unknown)}")
    keys = sorted(config.sweep.parameters)
    products = itertools.product(*(config.sweep.parameters[key] for key in keys)) if keys else [()]
    resolved = []
    for index, values in enumerate(products):
        overrides = dict(zip(keys, values, strict=True))
        resolved.append((index, apply_overrides(config, overrides), overrides))
    return resolved
