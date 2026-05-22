"""
Batch processing configuration for BookGap.

Defines batch-specific parameters like number of gaps to process,
completion windows, and output generation options.
"""

from dataclasses import dataclass

@dataclass
class BatchConfig:
    """Configuration for batch processing parameters.
    
    All parameters are required - use ConfigStore presets for common configurations.
    Controls batch job execution and OpenAI API integration.
    """
    
    # Processing scope - control how many research gaps to analyze in this batch
    num_gaps: int = 1  # Maximum number of gaps to process per topic (e.g., 2 = two gaps per topic)
    completion_window: str = "24h"  # Time window for batch completion ("24h" only supported option)
    generate_plots: bool = False     # Whether to create visualization plots for gaps (default: False)
    save_prompts: bool = True        # Whether to save individual prompt files for inspection (default: True)
    save_batch_file: bool = True     # Whether to save the JSONL batch request file (default: True)
    
    def __post_init__(self):
        """Validate batch configuration parameters."""
        if self.num_gaps <= 0:
            raise ValueError(f"num_gaps must be positive, got {self.num_gaps}")
            
        # Validate completion window (OpenAI only supports 24h currently)
        if self.completion_window != "24h":
            raise ValueError(f"completion_window must be '24h', got '{self.completion_window}'")
