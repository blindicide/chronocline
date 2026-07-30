"""Domain-specific exceptions."""


class ChronoclineError(Exception):
    """Base error for invalid models or numerical failures."""


class ValidationError(ChronoclineError):
    """Raised when a mathematical object violates its invariants."""


class UnsupportedScientificModelError(ChronoclineError):
    """Raised when a requested model lacks a defined scientific interpretation."""
