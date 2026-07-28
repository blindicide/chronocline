"""Logging setup for command-line experiments."""

from __future__ import annotations

import logging


def configure_logging(verbose: bool = False) -> None:
    """Configure concise, deterministic console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
