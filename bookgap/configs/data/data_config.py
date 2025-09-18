"""
Data source configuration for BookGap analysis.

Defines dataset parameters and deduplication settings.
"""

from dataclasses import dataclass
from typing import Optional

from bookgap.deduplicators import Deduplicator


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data source parameters.
    
    All parameters are required - use ConfigStore presets for common configurations.
    """
    
    # Dataset scope
    max_works: Optional[int]
    
    # Deduplication settings
    deduplicator: Optional[Deduplicator]
    
    def __post_init__(self):
        """Validate data configuration parameters."""
        if self.max_works is not None and self.max_works <= 0:
            raise ValueError(f"max_works must be positive, got {self.max_works}")
