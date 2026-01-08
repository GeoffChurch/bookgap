"""Reusable gap metric helpers for proposal evaluation.

These functions are side-effect free and intended for reuse by evaluators or
optimization loops that need to score candidate embeddings against known gaps.
"""

from typing import List

import numpy as np
from numpy.typing import NDArray

from bookgap.configs.config import ExperimentConfig
from bookgap.gap_statistics import compute_gap_statistics
from bookgap.llm.data_structures import GapWithContext, ProposalGapStats


def compute_gap_metrics_for_embeddings(
    gap: GapWithContext,
    embeddings: NDArray[np.float32],
    config: ExperimentConfig,
) -> List[ProposalGapStats]:
    """Compute gap metrics for arbitrary embeddings projected into a gap.

    Returns per-embedding gap size statistics without side effects so it can be
    reused by offline evaluators or prompt optimizers.
    """

    algorithm = config.gap.cell_type.value

    gap_sizes = algorithm.compute_gap_sizes_for_embeddings(
        ambient=gap.articles,
        landmark=gap.books,
        embeddings=embeddings,
        radius_fraction=config.gap.radius_fraction,
        k=1,
        is_intra_cloud=False,
    )

    book_gap_sizes = gap.book_gap_sizes

    results: List[ProposalGapStats] = []
    for gap_size in gap_sizes:
        percentile, zscore, _ = compute_gap_statistics(gap_size, book_gap_sizes)
        results.append(
            ProposalGapStats(
                gap_size=gap_size,
                gap_size_percentile=percentile,
                gap_zscore=zscore,
            )
        )

    return results
