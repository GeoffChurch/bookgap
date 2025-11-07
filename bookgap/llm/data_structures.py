"""Shared data structures for LLM workflow processing.

This module contains common data structures used across the LLM workflow
to avoid circular imports between different modules.
"""

from dataclasses import dataclass, field

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
    book_gap_sizes: list[int]  # Gap sizes for all books in topic (for statistics)
    gap_size: int
    gap_size_percentile: float
    gap_zscore: float
    shapiro_log10_pval: float


@dataclass(frozen=True)
class BatchResults:
    """Immutable container for batch processing results."""
    proposals: list[ParsedProposal]
    metadata: ExperimentConfig


@dataclass(frozen=True)
class ProposalOutrankCounts:
    """Stores, per in-gap article, how many books outrank a proposal."""

    counts: NDArray[np.int32]

    def coverage_fraction(self) -> float:
        num_covered = np.count_nonzero(self.counts == 0)
        return float(num_covered / self.counts.size)


@dataclass(frozen=True)
class ProposalGapStats:
    """Computed gap statistics for a proposal considered as a hypothetical book."""

    gap_size: int
    gap_size_percentile: float
    gap_zscore: float


@dataclass
class ProposalAnalytics:
    """Aggregated analytics captured for a single proposal."""

    gap_stats: ProposalGapStats | None = None
    outrank_counts: ProposalOutrankCounts | None = None

    def gap_coverage_fraction(self) -> float | None:
        return (
            self.outrank_counts.coverage_fraction()
            if self.outrank_counts is not None
            else None
        )


@dataclass
class ProposalAnalyticsStore:
    """Mutable container for analytics keyed by proposal custom ID."""

    _analytics: dict[str, ProposalAnalytics] = field(default_factory=dict)

    def _entry(self, proposal_custom_id: str) -> ProposalAnalytics:
        return self._analytics.setdefault(proposal_custom_id, ProposalAnalytics())

    def add_gap_stats(self, proposal_custom_id: str, stats: ProposalGapStats) -> None:
        self._entry(proposal_custom_id).gap_stats = stats

    def add_outrank_counts(
        self, proposal_custom_id: str, counts: ProposalOutrankCounts
    ) -> None:
        self._entry(proposal_custom_id).outrank_counts = counts

    def get(self, proposal_custom_id: str) -> ProposalAnalytics | None:
        return self._analytics.get(proposal_custom_id)

    def items(self):
        return self._analytics.items()
