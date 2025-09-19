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
    Controls batch job execution and OpenAI API integration.
    """
    
    # Processing scope - control how many research gaps to analyze in this batch
    num_gaps: int  # Total number of gaps to process across all topics (1-77, default: 1)
    
    # OpenAI batch API settings - control job execution timing and priority
    completion_window: str  # Time window for batch completion ("24h" only supported option)
    
    # Output generation - control what auxiliary files to generate during processing
    generate_plots: bool     # Whether to create visualization plots for gaps (default: True)
    save_prompts: bool       # Whether to save individual prompt files for inspection (default: True)
    save_batch_file: bool    # Whether to save the JSONL batch request file (default: True)
    
    def __post_init__(self):
        """Validate batch configuration parameters."""
        if self.num_gaps <= 0:
            raise ValueError(f"num_gaps must be positive, got {self.num_gaps}")
            
        # Validate completion window (OpenAI only supports 24h currently)
        if self.completion_window != "24h":
            raise ValueError(f"completion_window must be '24h', got '{self.completion_window}'")
