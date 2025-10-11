"""Shared data structures for LLM workflow processing.

This module contains common data structures used across the LLM workflow
to avoid circular imports between different modules.
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from bookgap.configs.config import ExperimentConfig
from bookgap.subcloud import SubCloud
from .protocols import GapProtocol, ParsedProposal


@dataclass(frozen=True)
class GapWithContext:
    """Gap with additional context for workflow processing."""
    gap: GapProtocol
    topic_name: str
    topic_id: str
    gap_idx: int
    books: SubCloud
    articles: SubCloud
    book_gap_sizes: List[int]  # Gap sizes for all books in topic (for statistics)
    gap_size: int
    gap_size_percentile: float
    gap_zscore: float
    shapiro_log10_pval: float


@dataclass(frozen=True)
class ArtifactPaths:
    """Container for all generated artifact file paths."""
    plots: List[Path]
    context_files: Optional[List[Path]] = None  # Will be added in Phase 2
    proposal_files: Optional[List[Path]] = None  # Will be added in Phase 2
    results_csv: Optional[Path] = None  # Will be added in Phase 2
    latex_files: Optional[List[Path]] = None  # Will be added in Phase 4


@dataclass(frozen=True)
class BatchResults:
    """Immutable container for batch processing results."""
    proposals: List[ParsedProposal]
    metadata: ExperimentConfig
    artifact_paths: Optional[ArtifactPaths] = None
