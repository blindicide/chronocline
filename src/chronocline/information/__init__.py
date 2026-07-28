"""Information-theoretic functions and solvers."""

from .blahut_arimoto import CapacityResult, blahut_arimoto
from .mutual_information import mutual_information, mutual_information_entropy

__all__ = ["CapacityResult", "blahut_arimoto", "mutual_information", "mutual_information_entropy"]
