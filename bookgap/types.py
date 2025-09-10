#!/usr/bin/env python3
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
