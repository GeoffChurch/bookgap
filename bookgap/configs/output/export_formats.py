"""
Export formats for BookGap output data.

Defines supported output formats for analysis results.
"""

from enum import Enum


class ExportFormat(Enum):
    """Output formats for analysis results."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    JSONL = "jsonl"
