"""
LLM-specific types and mappings.

This module contains LLM-specific algorithm mappings.
"""

from enum import Enum

from .gap_algorithms import KballAlgorithm, VoronoiAlgorithm


class CellTypeRegistry(Enum):
    """Registry mapping cell types to actual algorithm classes."""
    KBALL = KballAlgorithm
    VORONOI = VoronoiAlgorithm