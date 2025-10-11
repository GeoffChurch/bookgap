"""
Hydra structured configuration system for BookGap batch processing.

This module provides pure dataclass configurations for the clean batch-first
architecture, eliminating the need for AnalysisConfiguration bridge pattern.
"""

from .config import ExperimentConfig
from .gap.gap_config import GapConfig
from .llm.llm_config import LLMConfig
from .batch.batch_config import BatchConfig
from .output.output_config import OutputConfig
from .output.export_formats import ExportFormat
from .cloud.cloud_config import CloudConfig

__all__ = [
    "ExperimentConfig",
    "GapConfig", 
    "LLMConfig",
    "BatchConfig",
    "OutputConfig",
    "ExportFormat",
    "CloudConfig",
]
