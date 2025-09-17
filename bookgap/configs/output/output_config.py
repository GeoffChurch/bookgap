"""
Output configuration for BookGap analysis results.

Defines output formats, directory structure, and export options.
"""

from dataclasses import dataclass
from typing import Tuple

from .export_formats import ExportFormat


@dataclass(frozen=True)
class OutputConfig:
    """Configuration for output generation and export.
    
    All parameters are required - use ConfigStore presets for common configurations.
    """
    
    # Export formats
    export_formats: Tuple[ExportFormat, ...]
    
    # Output organization
    create_subdirectories: bool
    include_metadata: bool
    
    # File naming
    use_timestamp: bool
