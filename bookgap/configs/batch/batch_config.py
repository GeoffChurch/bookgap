"""
Batch processing configuration for BookGap.

Defines batch-specific parameters like number of gaps to process,
completion windows, and output generation options.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class BatchConfig:
    """Configuration for batch processing parameters.
    
    All parameters are required - use ConfigStore presets for common configurations.
    """
    
    # Processing scope
    num_gaps: int
    
    # OpenAI batch API settings
    completion_window: str
    
    # Output generation
    generate_plots: bool
    save_prompts: bool
    save_batch_file: bool
    
    def __post_init__(self):
        """Validate batch configuration parameters."""
        if self.num_gaps <= 0:
            raise ValueError(f"num_gaps must be positive, got {self.num_gaps}")
            
        # Validate completion window (OpenAI only supports 24h currently)
        if self.completion_window != "24h":
            raise ValueError(f"completion_window must be '24h', got '{self.completion_window}'")
