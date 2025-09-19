"""
Main experiment configuration for BookGap batch processing.

This is the top-level configuration class that combines all sub-configurations
and provides the primary interface for Hydra-managed experiments.
"""

from dataclasses import dataclass

from .gap.gap_config import GapConfig
from .llm.llm_config import LLMConfig
from .data.data_config import DataConfig
from .batch.batch_config import BatchConfig
from .output.output_config import OutputConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration for a complete BookGap batch experiment.
    
    All parameters are required - use ConfigStore presets for common configurations.
    Combines all sub-configurations into a complete experiment specification.
    """
    
    # Core configuration components - each controls a different aspect of the analysis pipeline
    gap: GapConfig      # Gap detection algorithm parameters and spatial search settings
    llm: LLMConfig      # Language model configuration, prompt structure, and generation settings  
    data: DataConfig    # Data source parameters, dataset scope, and deduplication strategy
    batch: BatchConfig  # Batch processing control, OpenAI API settings, and job execution options
    output: OutputConfig # Output formatting, export options, and file organization preferences
