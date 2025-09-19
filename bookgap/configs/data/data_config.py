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
    Controls dataset scope and deduplication strategy for OpenAlex research data.
    """
    
    # Dataset scope - limit data volume for development and testing
    max_works: Optional[int]  # Maximum papers/books to fetch (None=unlimited, useful for debugging)
    
    # Deduplication settings - remove semantically similar papers to improve gap quality
    deduplicator: Optional[Deduplicator]  # Strategy for removing duplicate works:
                                         # - SqDistDeduplicator(min_sqdist=0.01): Fast range-based dedup
                                         # - FractionalDeduplicator(fraction=0.1): Percentile-based dedup
                                         # - None: No deduplication
    
    def __post_init__(self):
        """Validate data configuration parameters."""
        if self.max_works is not None and self.max_works <= 0:
            raise ValueError(f"max_works must be positive, got {self.max_works}")
