"""Gap detection configuration."""

from dataclasses import dataclass

from bookgap.types import CellType
@dataclass(frozen=True)
class GapConfig:
    """Configuration for gap detection parameters.

    All parameters are required - use ConfigStore presets for common configurations.
    Compatible with existing GapConfiguration in config_manager.py.
    """

    # Gap detection algorithm - determines which spatial algorithm to use
    cell_type: CellType = CellType.KBALL  # KBALL or VORONOI cell algorithm for finding research gaps

    # Spatial parameters - control gap detection sensitivity and coverage
    radius_fraction: float = 0.75  # Fraction of k-NN distance for gap radius (0.0-1.0, default: 0.75)

