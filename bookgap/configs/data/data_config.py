"""
Data source configuration for BookGap analysis.

Defines dataset parameters and deduplication settings.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SqDistDeduplicatorConfig:
    """Configuration for SqDistDeduplicator.
    
    Removes semantically similar works based on squared embedding distance threshold.
    Fast range-based deduplication suitable for large datasets.
    """
    
    _target_: str = field(default="bookgap.deduplicators.SqDistDeduplicator", init=False)
    min_sqdist: float  # Minimum squared distance threshold (e.g., 0.01 = 1% distance)
    
    def __post_init__(self):
        """Validate SqDistDeduplicator configuration parameters."""
        if self.min_sqdist <= 0:
            raise ValueError(f"min_sqdist must be positive, got {self.min_sqdist}")


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data source parameters.
    
    All parameters are required - use ConfigStore presets for common configurations.
    Controls dataset scope and deduplication strategy for OpenAlex research data.
    """
    
    # Dataset scope - limit data volume for development and testing
    max_works: Optional[int]  # Maximum papers/books to fetch (None=unlimited, useful for debugging)
    
    # Deduplication settings - remove semantically similar papers to improve gap quality
    deduplicator: Optional[SqDistDeduplicatorConfig]  # Strategy for removing duplicate works:
                                                     # - SqDistDeduplicatorConfig: Fast range-based dedup
                                                     # - None: No deduplication
    
    def __post_init__(self):
        """Validate data configuration parameters."""
        if self.max_works is not None and self.max_works <= 0:
            raise ValueError(f"max_works must be positive, got {self.max_works}")
