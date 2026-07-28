"""Quantization models and compatibility checks."""

from .uniform import UniformQuantizer


def is_nested(fine: UniformQuantizer, coarse: UniformQuantizer, tolerance: float = 1e-12) -> bool:
    """Return whether compatible floor quantizers permit deterministic coarsening."""
    ratio = coarse.step / fine.step
    return (
        fine.mode == coarse.mode == "floor"
        and abs(ratio - round(ratio)) <= tolerance
        and abs(
            (coarse.phase - fine.phase) / fine.step - round((coarse.phase - fine.phase) / fine.step)
        )
        <= tolerance
    )


__all__ = ["UniformQuantizer", "is_nested"]
