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
    Controls how analysis results are formatted and organized for downstream use.
    """
    
    # Export formats - specify which file formats to generate for results
    export_formats: Tuple[ExportFormat, ...]  # Combination of CSV, JSON, JSONL, PARQUET formats
    
    # Output organization - control directory structure and file organization
    create_subdirectories: bool  # Whether to organize outputs into topic/gap subdirectories (default: True)
    include_metadata: bool       # Whether to include configuration metadata in output files (default: True)
    
    # File naming - control how output files are named and timestamped
    use_timestamp: bool  # Whether to include timestamps in output filenames (default: True)
