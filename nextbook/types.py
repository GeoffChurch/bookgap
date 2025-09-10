#!/usr/bin/env python3
"""
Core types and enums for nextbook analysis.

This module contains fundamental types used across the codebase,
moved here to avoid circular dependencies.
"""

from enum import Enum

from nextbook.llm.gap_algorithms.kball_algorithm import KballAlgorithm
from nextbook.llm.gap_algorithms.voronoi_algorithm import VoronoiAlgorithm


class CellType(Enum):
    """Types of cells used in gap analysis algorithms."""
    KBALL = KballAlgorithm
    VORONOI = VoronoiAlgorithm
