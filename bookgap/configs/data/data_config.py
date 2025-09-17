"""
Data source configuration for BookGap analysis.

Defines dataset parameters, work types, filtering options, and deduplication settings.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from bookgap.util import WorkType, SubField, Topic, Filters, Filter
from bookgap.cloud import WorkFilter
from bookgap.deduplicators import Deduplicator


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data source and filtering parameters.
    
    All parameters are required - use ConfigStore presets for common configurations.
    """
    
    # Dataset scope
    subfield: SubField
    max_works: Optional[int]
    work_types: Tuple[WorkType, ...]
    
    # Filtering configuration  
    work_filters: Tuple[WorkFilter, ...]
    filter_preset: Optional[Filters]
    custom_filter: Optional[Filter]
    
    # Deduplication settings
    deduplicator: Optional[Deduplicator]
    
    def __post_init__(self):
        """Validate data configuration parameters."""
        if self.max_works is not None and self.max_works <= 0:
            raise ValueError(f"max_works must be positive, got {self.max_works}")
            
        if not self.work_types:
            raise ValueError("work_types cannot be empty")
            
        # Validate work_types are unique
        if len(set(self.work_types)) != len(self.work_types):
            raise ValueError("work_types must be unique")
            
        # Validate filter consistency
        if self.filter_preset is not None and self.custom_filter is not None:
            raise ValueError("Cannot specify both filter_preset and custom_filter")
