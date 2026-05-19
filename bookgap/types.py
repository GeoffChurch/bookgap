"""
Core types and enums for bookgap analysis.

This module contains fundamental types used across the codebase,
moved here to avoid circular dependencies.
"""

from enum import Enum

from bookgap.llm.gap_algorithms.kball_algorithm import KballAlgorithm
from bookgap.llm.gap_algorithms.voronoi_algorithm import VoronoiAlgorithm


class CellType(Enum):
    """Types of cells used in gap analysis algorithms."""
    KBALL = KballAlgorithm
    VORONOI = VoronoiAlgorithm


class DeflationStrategy(Enum):
    """Gap-deflation strategy selectable on the K-ball path.

    GREEDY_HARD is the byte-identical production default (the existing
    greedy hard-subtraction pruning). FL_BOOKDEMAND_RESID is the
    opt-in facility-location-over-book∪orphan-background deflation
    validated in the gap-deflation bake-off. Voronoi ignores this.
    """
    GREEDY_HARD = "greedy_hard"
    FL_BOOKDEMAND_RESID = "fl_bookdemand_resid"
