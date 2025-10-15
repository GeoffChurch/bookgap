"""Shared data structures for LLM workflow processing.

This module contains common data structures used across the LLM workflow
to avoid circular imports between different modules.
"""

from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

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
class BatchResults:
    """Immutable container for batch processing results."""
    proposals: List[ParsedProposal]
    metadata: ExperimentConfig


@dataclass(frozen=True)
class ProposalOutrankCounts:
    """Stores, per in-gap article, how many books outrank a proposal."""

    counts: NDArray[np.int32]

    def coverage_fraction(self) -> float:
        num_covered = np.count_nonzero(self.counts == 0)
        return float(num_covered / self.counts.size)


ProposalOutrankCountsMap = Dict[
    Tuple[str, int], Dict[str, Deque[ProposalOutrankCounts]]
]
