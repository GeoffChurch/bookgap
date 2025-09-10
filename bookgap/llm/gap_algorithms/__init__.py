"""
Gap algorithms package.

This package contains implementations of different gap finding algorithms,
structured as classes that can be cleanly used in enums.
"""

from .kball_algorithm import KballAlgorithm
from .voronoi_algorithm import VoronoiAlgorithm

__all__ = ['KballAlgorithm', 'VoronoiAlgorithm']
