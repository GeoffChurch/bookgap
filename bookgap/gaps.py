"""Shared helpers for gap discovery and caching.

This module provides side-effect-free gap computation and simple cache helpers
so that both batch workflows and optimization scripts can reuse the same logic
without importing generator-specific code.
"""

from __future__ import annotations

from itertools import islice
from typing import Any, Optional

from bookgap.pyalex_client import pyalex

from hydra.utils import instantiate

from bookgap.cloud import Cloud
from bookgap.corpus import Corpus
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
    landmark, ambient = get_subclouds_for_topic(cloud, topic_id)  # defaults book/article
    return detect_gaps(
        ambient=ambient, landmark=landmark, query=ambient,
        cell_type=cell_type, num_gaps=num_gaps, radius_fraction=radius_fraction,
        deflation_strategy=deflation_strategy, size_metric=size_metric,
        candidate_pool_size=candidate_pool_size, label_id=topic_id, label_name=topic_name,
    )


def detect_gaps(
    ambient: Corpus,
    landmark: Corpus,
    *,
    query: Corpus | None = None,
    cell_type: CellType,
    num_gaps: int,
    radius_fraction: float,
    deflation_strategy: DeflationStrategy = DeflationStrategy.GREEDY_HARD,
    size_metric: str = "raw_kball",
    candidate_pool_size: int = 10,
    label_id: str = "",
    label_name: str = "",
) -> list[GapWithContext]:
    """Source-agnostic gap detection: gaps in `landmark` w.r.t. `ambient` density.

    `query` (default = `ambient`) is the set of points evaluated for gap-membership.
    `label_id`/`label_name` are opaque labels stored on the result (the OpenAlex
    sweep fills them with topic id/name). Requires ambient/landmark/query share an
    embedder (shared embedding space).
    """
    query = query if query is not None else ambient
    if not (ambient.embedder_fingerprint == landmark.embedder_fingerprint == query.embedder_fingerprint):
        raise ValueError("ambient/landmark/query must share one embedder (shared embedding space)")

    num_landmark = len(landmark.embeddings())
    num_ambient = len(ambient.embeddings())
    LOGGER.info("%s: %s landmark, %s ambient", label_name or label_id, num_landmark, num_ambient)
    if num_ambient == 0:
        LOGGER.warning("No ambient points for %s", label_id)
        return []

    landmark_embeddings = landmark.embeddings()
    book_gap_sizes = cell_type.value.compute_gap_sizes_for_embeddings(
        ambient=ambient, landmark=landmark, embeddings=landmark_embeddings,
        radius_fraction=radius_fraction, k=1, is_intra_cloud=True,
    )
    if len(book_gap_sizes) != len(landmark_embeddings):
        raise ValueError(
            f"book_gap_sizes length mismatch: {len(book_gap_sizes)} vs {len(landmark_embeddings)} for {label_name}"
        )

    gap_generator = cell_type.value.find_gaps(
        ambient=ambient, landmark=landmark, query=query,
        radius_fraction=radius_fraction, k=1, is_intra_cloud=False,
        deflation_strategy=deflation_strategy,
    )

    if size_metric == "raw_kball":
        ordered_gaps = list(islice(gap_generator, num_gaps))
    else:
        metric_fn = gap_size_metrics.get(size_metric)
        pool = list(islice(gap_generator, max(num_gaps, candidate_pool_size)))
        ambient_emb = np.ascontiguousarray(ambient.embeddings(), dtype=np.float32)
        landmark_emb_arr = np.ascontiguousarray(landmark_embeddings, dtype=np.float32)
        scored = []
        for gap in pool:
            members = np.asarray(gap.get_dists_idxs().idxs, dtype=np.int64)
            scored.append((metric_fn(members, ambient_emb, landmark_emb_arr), gap))
        scored.sort(key=lambda sg: sg[0], reverse=True)
        ordered_gaps = [g for _, g in scored[:num_gaps]]

    out: list[GapWithContext] = []
    for i, gap in enumerate(ordered_gaps):
        gap_size = len(gap.get_dists_idxs().idxs)
        pct, z, shap = compute_gap_statistics(gap_size, book_gap_sizes)
        out.append(GapWithContext(
            gap=gap, topic_name=label_name, topic_id=label_id, gap_idx=i,
            books=landmark, articles=ambient, book_gap_sizes=book_gap_sizes,
            gap_size=gap_size, gap_size_percentile=pct, gap_zscore=z, shapiro_log10_pval=shap,
        ))
    if not out:
        LOGGER.warning("No gaps found for %s", label_name)
    return out


@MEMORY.cache
def _compute_gaps_impl(
    cloud: Cloud,
    radius_fraction: float,
    cell_type: CellType,
    num_gaps: int,
    deflation_strategy: DeflationStrategy = DeflationStrategy.GREEDY_HARD,
    size_metric: str = "raw_kball",
    candidate_pool_size: int = 10,
    topic_subset: Optional[tuple] = None,
) -> list[GapWithContext]:
    """Internal implementation that computes gaps for a single configuration.

    Cached by joblib.Memory using the hashable parameters (including
    ``deflation_strategy`` and ``topic_subset``) as the cache key.
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

    if topic_subset:
        subset_set = set(topic_subset)
        filtered = [t for t in valid_topics if t in subset_set]
        missing = subset_set - set(filtered)
        if missing:
            LOGGER.warning(
                "topic_subset includes %d topic(s) not resolved from filter: %s",
                len(missing), sorted(missing),
            )
        LOGGER.info(
            "topic_subset restricts gap-detection to %d/%d topics",
            len(filtered), len(valid_topics),
        )
        valid_topics = filtered

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
    topic_subset = getattr(config.gap, "topic_subset", None)
    return _compute_gaps_impl(
        cloud=cloud,
        radius_fraction=config.gap.radius_fraction,
        cell_type=config.gap.cell_type,
        num_gaps=config.batch.num_gaps,
        deflation_strategy=config.gap.deflation_strategy,
        size_metric=getattr(config.gap, "size_metric", "raw_kball"),
        candidate_pool_size=getattr(config.gap, "candidate_pool_size", 10),
        topic_subset=tuple(topic_subset) if topic_subset else None,
    )
