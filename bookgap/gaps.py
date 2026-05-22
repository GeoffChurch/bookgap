"""Shared helpers for gap discovery and caching.

This module provides side-effect-free gap computation and simple cache helpers
so that both batch workflows and optimization scripts can reuse the same logic
without importing generator-specific code.
"""

from __future__ import annotations

from itertools import islice
from typing import Any

from bookgap.pyalex_client import pyalex

from hydra.utils import instantiate

from bookgap.cloud import Cloud
from bookgap.openalex_scopes import resolve_filter_upward, get_subfield_topics, get_field_topics
from bookgap.configs.config import ExperimentConfig
from bookgap.gap_statistics import compute_gap_statistics
from bookgap import gap_size_metrics
from bookgap.llm.data_structures import GapWithContext
from bookgap.subcloud import get_subclouds_for_topic
from bookgap.types import CellType, DeflationStrategy
from bookgap.util import LOGGER, MEMORY

import numpy as np


@MEMORY.cache
def _compute_gaps_for_topic(
    cloud: Cloud,
    topic_id: str,
    topic_name: str,
    radius_fraction: float,
    cell_type: CellType,
    num_gaps: int,
    deflation_strategy: DeflationStrategy = DeflationStrategy.GREEDY_HARD,
    size_metric: str = "raw_kball",
    candidate_pool_size: int = 10,
) -> list[GapWithContext]:
    """Compute gaps for a single topic (cached per-topic).

    Cached by joblib.Memory using the topic parameters as the cache key.
    ``deflation_strategy``, ``size_metric``, and ``candidate_pool_size``
    are explicit arguments (not derived from a config object joblib can't
    see into) so a config change produces a different cache key.

    When ``size_metric != "raw_kball"``: consume ``candidate_pool_size``
    candidates from the gap generator, score each by the chosen metric,
    sort by score desc, take top ``num_gaps``. When ``size_metric ==
    "raw_kball"``: take the first ``num_gaps`` candidates from the
    generator (byte-identical to historical behaviour).
    """
    books, articles = get_subclouds_for_topic(cloud, topic_id)
    num_books = len(books.embeddings())
    num_articles = len(articles.embeddings())
    LOGGER.info("Topic %s: %s books, %s articles", topic_name, num_books, num_articles)

    if num_articles == 0:
        LOGGER.warning("No articles found for topic %s", topic_id)
        return []

    # Compute per-book gap sizes using the algorithm's explicit size helper to ensure
    # one gap size per book (find_gaps can drop books with zero coverage, causing length mismatches).
    book_embeddings = books.embeddings()
    # Single call: k-ball requires is_intra_cloud=True (query==landmark so drop self-neighbor);
    # Voronoi ignores is_intra_cloud, so keeping it True is harmless and simplifies branching.
    book_gap_sizes = cell_type.value.compute_gap_sizes_for_embeddings(
        ambient=articles,
        landmark=books,
        embeddings=book_embeddings,
        radius_fraction=radius_fraction,
        k=1,
        is_intra_cloud=True,
    )

    if len(book_gap_sizes) != len(book_embeddings):
        raise ValueError(
            "book_gap_sizes length mismatch: "
            f"{len(book_gap_sizes)} sizes vs {len(book_embeddings)} books for topic {topic_name}"
        )

    # Find article-query gaps with configured parameters
    gap_generator = cell_type.value.find_gaps(
        ambient=articles,
        landmark=books,
        query=articles,
        radius_fraction=radius_fraction,
        k=1,
        is_intra_cloud=False,
        deflation_strategy=deflation_strategy,
    )

    # Re-rank by size_metric if it isn't the historical default. raw_kball
    # is the no-op: the generator already yields candidates by descending
    # raw size (greedy_hard) or in FL_BOOKDEMAND_RESID's own order.
    if size_metric == "raw_kball":
        ordered_gaps = list(islice(gap_generator, num_gaps))
    else:
        metric_fn = gap_size_metrics.get(size_metric)
        pool_size = max(num_gaps, candidate_pool_size)
        pool = list(islice(gap_generator, pool_size))
        article_emb = np.ascontiguousarray(articles.embeddings(), dtype=np.float32)
        book_emb_arr = np.ascontiguousarray(book_embeddings, dtype=np.float32)
        scored = []
        for gap in pool:
            members = np.asarray(gap.get_dists_idxs().idxs, dtype=np.int64)
            score = metric_fn(members, article_emb, book_emb_arr)
            scored.append((score, gap))
        # Sort by score desc; stable so that ties preserve generator order.
        scored.sort(key=lambda sg: sg[0], reverse=True)
        ordered_gaps = [g for _, g in scored[:num_gaps]]
        LOGGER.info(
            "Topic %s: re-ranked %d candidates by %s; "
            "top score=%.1f (raw_size=%d), bottom=%.1f (raw_size=%d)",
            topic_name, len(scored), size_metric,
            scored[0][0], len(scored[0][1].get_dists_idxs().idxs),
            scored[-1][0], len(scored[-1][1].get_dists_idxs().idxs),
        )

    topic_gaps: list[GapWithContext] = []
    for i, gap in enumerate(ordered_gaps):
        gap_size = len(gap.get_dists_idxs().idxs)
        LOGGER.info("Found gap of size %s articles", gap_size)
        gap_size_percentile, gap_zscore, shapiro_log10_pval = compute_gap_statistics(
            gap_size, book_gap_sizes
        )

        topic_gaps.append(
            GapWithContext(
                gap=gap,
                topic_name=topic_name,
                topic_id=topic_id,
                gap_idx=i,
                books=books,
                articles=articles,
                book_gap_sizes=book_gap_sizes,
                gap_size=gap_size,
                gap_size_percentile=gap_size_percentile,
                gap_zscore=gap_zscore,
                shapiro_log10_pval=shapiro_log10_pval,
            )
        )

    if not topic_gaps:
        LOGGER.warning("No gaps found for topic %s", topic_name)

    return topic_gaps

@MEMORY.cache
def _compute_gaps_impl(
    cloud: Cloud,
    radius_fraction: float,
    cell_type: CellType,
    num_gaps: int,
    deflation_strategy: DeflationStrategy = DeflationStrategy.GREEDY_HARD,
    size_metric: str = "raw_kball",
    candidate_pool_size: int = 10,
) -> list[GapWithContext]:
    """Internal implementation that computes gaps for a single configuration.

    Cached by joblib.Memory using the hashable parameters (including
    ``deflation_strategy``) as the cache key.
    """

    # Merge pre and post filters to get the most specific constraints
    merged_filter = cloud.filter_pair.merged()
    
    # Resolve upward to fill in missing field/subfield, pinned to the cloud's snapshot.
    resolved_filter = resolve_filter_upward(merged_filter, snapshot_date=cloud.snapshot_date)

    # Determine which topics to iterate over
    if resolved_filter.topic is not None:
        # Single topic specified
        valid_topics = [resolved_filter.topic]
    elif resolved_filter.subfield is not None:
        # Subfield specified: enumerate topics
        field_id = resolved_filter.field
        subfield_id = resolved_filter.subfield
        assert field_id is not None
        valid_topics = get_subfield_topics(subfield_id, snapshot_date=cloud.snapshot_date)
    else:
        # Field-only: enumerate all topics for this field
        field_id = resolved_filter.field
        assert field_id is not None
        valid_topics = get_field_topics(field_id, snapshot_date=cloud.snapshot_date)
    
    if not valid_topics:
        raise ValueError("No topics resolved from cloud.filter_pair.merged()")
    
    gaps: list[GapWithContext] = []

    from bookgap.snapshot_ingest import load_taxonomy
    pinned_tax = load_taxonomy(cloud.snapshot_date)

    for i, topic_id in enumerate(valid_topics):
        # Topic metadata for display, drawn from the snapshot taxonomy.
        topic_rec = pinned_tax.topics.get(topic_id, {})
        topic_name = topic_rec.get("display_name", topic_id)

        LOGGER.info(
            "Analyzing topic %s/%s: %s (%s)",
            i + 1,
            len(valid_topics),
            topic_name,
            topic_id,
        )
        gaps.extend(
            _compute_gaps_for_topic(
                cloud=cloud,
                topic_id=topic_id,
                topic_name=topic_name,
                radius_fraction=radius_fraction,
                cell_type=cell_type,
                num_gaps=num_gaps,
                deflation_strategy=deflation_strategy,
                size_metric=size_metric,
                candidate_pool_size=candidate_pool_size,
            )
        )

    LOGGER.info("Found %s gaps total", len(gaps))
    return gaps


def compute_gaps_for_config(config: ExperimentConfig) -> list[GapWithContext]:
    """Compute gaps for all topics defined by the configured filter (cached).
    
    Extracts hashable parameters from config and delegates to cached implementation.
    """
    cloud: Cloud = instantiate(config.cloud)
    return _compute_gaps_impl(
        cloud=cloud,
        radius_fraction=config.gap.radius_fraction,
        cell_type=config.gap.cell_type,
        num_gaps=config.batch.num_gaps,
        deflation_strategy=config.gap.deflation_strategy,
        size_metric=getattr(config.gap, "size_metric", "raw_kball"),
        candidate_pool_size=getattr(config.gap, "candidate_pool_size", 10),
    )
