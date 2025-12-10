"""Configuration for historical gap analysis."""

from dataclasses import dataclass, field
from typing import Optional

from bookgap.configs.config import ExperimentConfig
from bookgap.configs.gap.gap_config import GapConfig
from bookgap.configs.llm.llm_config import LLMConfig
from bookgap.configs.batch.batch_config import BatchConfig
from bookgap.configs.output.output_config import OutputConfig
from bookgap.configs.cloud.cloud_config import CloudConfig


@dataclass
class HistoricalAnalysisConfig:
    """Configuration for historical gap analysis.
    
    Analyzes how books published in year+1 fill gaps in the embedding space
    as of the end of `year`.
    """
    
    year: int
    """Year to use as temporal cutoff (books up to this year form the landmark set)."""
    
    # Provide a safe default experiment composed of default sub-configs
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(
            gap=GapConfig(),
            llm=LLMConfig(),
            batch=BatchConfig(),
            output=OutputConfig(),
            cloud=CloudConfig(),
        )
    )
    """Experiment configuration (gap detection, cloud, LLM, batch, output)."""
    
    min_books_per_year: int = 10
    """Minimum number of books required in year+1 to include in analysis."""
    
    min_articles_per_year: int = 10
    """Minimum number of articles required in year+1 to include in analysis."""
    
    output_dir: Optional[str] = None
    """Output directory for results CSV (defaults to artifacts_scripts/historical_gap_analysis/results/)."""
