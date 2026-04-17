"""
Main experiment configuration for BookGap batch processing.

This is the top-level configuration class that combines all sub-configurations
and provides the primary interface for Hydra-managed experiments.
"""

from dataclasses import dataclass, field
from typing import Optional

from hydra.utils import instantiate

from bookgap.cloud import Cloud

from .gap.gap_config import GapConfig
from .llm.llm_config import LLMConfig
from .batch.batch_config import BatchConfig
from .output.output_config import OutputConfig
from .cloud.cloud_config import CloudConfig


@dataclass
class SnapshotConfig:
    """Configuration for snapshot-based ingestion from the OpenAlex S3 snapshot."""

    snapshot_date: str = "2025-11-06"  # override per run; pins reproducibility
    bucket: str = "openalex"
    prefix: str = "data/works/"
    shard_row_flush_threshold: int = 10_000
    max_resident_rows: int = 1_000_000
    disk_abort_pct: float = 90.0


@dataclass
class ExperimentConfig:
    """Top-level configuration for a complete BookGap batch experiment.
    
    All parameters are required - use ConfigStore presets for common configurations.
    Combines all sub-configurations into a complete experiment specification.
    """
    
    # Core configuration components - each controls a different aspect of the analysis pipeline
    gap: GapConfig      # Gap detection algorithm parameters and spatial search settings
    llm: LLMConfig      # Language model configuration, prompt structure, and generation settings
    batch: BatchConfig  # Batch processing control, OpenAI API settings, and job execution options
    output: OutputConfig # Output formatting, export options, and file organization preferences
    cloud: CloudConfig  # Cloud construction parameters for gap analysis workflows

    # Optional snapshot ingestion config; None means use the default API path
    snapshot: Optional[SnapshotConfig] = field(default=None)
    
    def __post_init__(self):
        """Validate that gap work types are compatible with cloud filter."""
        
        # Instantiate cloud to get the actual FilterPair
        cloud_inst: Cloud = instantiate(self.cloud)
        merged_filter = cloud_inst.filter_pair.merged()
        
        if merged_filter.work_types is None:
            raise ValueError(
                "cloud.filter_pair merged filter must specify work_types"
            )
        
        work_types_set = set(merged_filter.work_types)
        
        if (missing := {self.gap.ambient_work_type, self.gap.landmark_work_type} - work_types_set):
            raise ValueError(
                f"gap work types {missing} not in "
                f"cloud.filter_pair.merged().work_types ({work_types_set})"
            )
        
