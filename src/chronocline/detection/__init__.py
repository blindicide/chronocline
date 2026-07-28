"""Statistical timing-distribution detection."""

from .empirical import detection_trial
from .likelihood_ratio import log_likelihood_ratio

__all__ = ["detection_trial", "log_likelihood_ratio"]
