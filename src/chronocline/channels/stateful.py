"""Markers for stateful channels; exact memoryless capacity is intentionally unavailable."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StatefulChannelNotice:
    """Metadata preventing accidental exact-memoryless claims."""

    estimator: str
    stateful: bool = True
