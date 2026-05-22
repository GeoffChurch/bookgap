"""
Output configuration for BookGap analysis results.

Defines output formats, directory structure, and export options.
"""

from dataclasses import dataclass
from typing import Tuple

from .export_formats import ExportFormat


@dataclass
class OutputConfig:
    """Configuration for output generation and export.
    
    All parameters are required - use ConfigStore presets for common configurations.
    Controls how analysis results are formatted and organized for downstream use.
    """
    
    # Export formats - specify which file formats to generate for results
    export_formats: Tuple[ExportFormat, ...] = (ExportFormat.CSV, ExportFormat.JSON, ExportFormat.JSONL, ExportFormat.PARQUET)
    create_subdirectories: bool = True
    include_metadata: bool = True
    use_timestamp: bool = True
