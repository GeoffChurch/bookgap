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
    """
    
    # Core configuration components
    gap: GapConfig
    llm: LLMConfig
    data: DataConfig
    batch: BatchConfig
    output: OutputConfig
    
    # Experiment metadata
    experiment_name: str
    description: str = "BookGap batch analysis experiment"
    
    def __post_init__(self):
        """Validate overall experiment configuration."""
        # Cross-component validation can go here
        # For example, ensuring LLM chapter constraints are reasonable for the dataset
        pass
