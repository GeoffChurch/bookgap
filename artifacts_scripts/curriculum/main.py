#!/usr/bin/env python3
"""Unified curriculum building pipeline.

This is the one-stop-shop script for building curricula. It orchestrates:
1. Curriculum building (PPR + hierarchical clustering)
2. Keyword extraction for topic summarization
3. Visualization of the curriculum sequence

All steps can be toggled via Hydra configuration.

On the Algebra of Logic (1885): W2325835442
Attention is All You Need: W2626778328
LSTM: W2064675550
LoRA for Audio: W4391021820

Usage:
    python -m artifacts_scripts.curriculum.main \
        "target_work_ids=['https://openalex.org/W4391021820']" \
        desired_n_topics=5 \
        output_dir=results/lstm
        
    # Use different keyword extraction method
    python -m artifacts_scripts.curriculum.main \
        "target_work_ids=['https://openalex.org/W4391021820']" \
        keywords.method=log_odds \
        keywords.top_k=15
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import scipy.sparse as sp

from bookgap.cloud import Cloud
from bookgap.configs.config_store import register_configs
from bookgap.util import LOGGER, setup_logging

from .config import (
    CurriculumConfig,
    KeywordConfig,
    PlotConfig,
    get_curriculum_output_dir,
    derive_output_dir_name,
)
from .build_curriculum import run_curriculum_building
from .summarize_topics import summarize_topics, load_curriculum_outputs
from .plot_curriculum import plot_curriculum
from .output_summary import generate_curriculum_summary


# Register configs
register_configs()

# Register curriculum config
cs = ConfigStore.instance()
cs.store(name="curriculum", node=CurriculumConfig())


def _apply_dynamic_topk(
    keywords: Dict[int, List[tuple[str, float]]],
    *,
    floor: float,
    min_k: int,
) -> Dict[int, List[tuple[str, float]]]:
    """Trim each topic to keywords scoring >= ``floor`` * (topic's top score).

    Keywords arrive sorted descending by score. A healthy chapter's scores decay
    smoothly so all clear the relative floor; a thin/degenerate chapter's scores
    cliff to ~0 after a few real terms, so only those survive (no ~0-score junk
    padding the fixed top_k). Always keeps at least ``min_k``. Report only as many
    keywords as the evidence supports.
    """
    trimmed: Dict[int, List[tuple[str, float]]] = {}
    for tidx, kws in keywords.items():
        if not kws:
            trimmed[tidx] = kws
            continue
        top = max((s for _, s in kws), default=0.0)
        if top <= 0:
            trimmed[tidx] = kws[:min_k]
            continue
        cut = [(w, s) for w, s in kws if s >= floor * top]
        trimmed[tidx] = cut if len(cut) >= min_k else kws[:min_k]
    n_before = sum(len(v) for v in keywords.values())
    n_after = sum(len(v) for v in trimmed.values())
    LOGGER.info(
        "Dynamic top-k (floor=%.3f, min=%d): %d -> %d keywords across %d topics "
        "(sizes: %s)",
        floor, min_k, n_before, n_after, len(trimmed),
        ", ".join(str(len(trimmed[t])) for t in sorted(trimmed)),
    )
    return trimmed


def run_keyword_extraction(
    cfg: KeywordConfig,
    curriculum_dir: Path,
    cloud: Cloud,
    filtered_indices: Optional[np.ndarray],
) -> Optional[Dict[int, List[tuple[str, float]]]]:
    """Run keyword extraction for curriculum topics.
    
    Args:
        cfg: Keyword extraction configuration.
        curriculum_dir: Path to curriculum outputs directory.
        cloud: Cloud object for accessing works.
        filtered_indices: Array of indices into cloud if filtered, else None.
        
    Returns:
        Dictionary mapping topic index to list of (keyword, score) tuples,
        or None if extraction is disabled.
    """
    if not cfg.enabled:
        LOGGER.info("Keyword extraction disabled, skipping...")
        return None
    
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("KEYWORD EXTRACTION")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Method: {cfg.method}")
    LOGGER.info(f"Top-k keywords: {cfg.top_k}")
    LOGGER.info(f"Samples per topic: {cfg.samples_per_topic}")
    LOGGER.info(f"Doc weighting: {cfg.doc_weighting}")
    LOGGER.info(f"Topic weighting: {cfg.topic_weighting}")
    LOGGER.info(f"Background mode: {cfg.background_mode}")
    LOGGER.info(f"Weight by freq: {cfg.weight_by_freq}")
    LOGGER.info(
        f"Phrase mining: enabled={cfg.bigrams_enabled}, "
        f"npmi>={cfg.bigrams_npmi_threshold}, "
        f"be>={cfg.bigrams_min_boundary_be} bits / q={cfg.bigrams_be_quantile}, "
        f"normalize_be={cfg.bigrams_normalize_be}, "
        f"count>={cfg.bigrams_count_floor}, "
        f"max_per_iter={cfg.bigrams_max}, "
        f"iterations<={cfg.bigram_iterations}"
    )
    
    all_texts = [f"{t}\n{a}" for t, a in zip(cloud.work_titles(), cloud.work_abstracts())]
    if filtered_indices is not None:
        works_for_keywords = [all_texts[i] for i in filtered_indices]
    else:
        works_for_keywords = all_texts

    embeddings_for_keywords = None
    citation_adj_for_keywords = None
    if cfg.soft_topics:
        LOGGER.info(f"Soft topics enabled (mode={cfg.soft_mode}, floor={cfg.soft_floor})")
        if cfg.soft_mode == "citation":
            from bookgap.cloud import get_cloud_citation_graph
            adj = get_cloud_citation_graph(cloud, normalize=False)
            citation_adj_for_keywords = (
                adj[filtered_indices][:, filtered_indices] if filtered_indices is not None else adj
            )
        else:
            emb = np.asarray(cloud.embeddings())
            embeddings_for_keywords = emb[filtered_indices] if filtered_indices is not None else emb

    keywords = summarize_topics(
        curriculum_dir=curriculum_dir,
        texts=works_for_keywords,
        embeddings=embeddings_for_keywords,
        soft_topics=cfg.soft_topics,
        soft_mode=cfg.soft_mode,
        soft_perplexity=cfg.soft_perplexity,
        soft_alpha=cfg.soft_alpha,
        soft_alpha_mode=cfg.soft_alpha_mode,
        soft_class_balance=cfg.soft_class_balance,
        soft_floor=cfg.soft_floor,
        citation_adj=citation_adj_for_keywords,
        samples_per_topic=cfg.samples_per_topic,
        top_k_keywords=cfg.top_k,
        method=cfg.method,
        doc_weighting=cfg.doc_weighting,
        topic_weighting=cfg.topic_weighting,
        background_mode=cfg.background_mode,
        weight_by_freq=cfg.weight_by_freq,
        bigrams_enabled=cfg.bigrams_enabled,
        bigrams_npmi_threshold=cfg.bigrams_npmi_threshold,
        bigrams_max=cfg.bigrams_max,
        bigrams_min_boundary_be=cfg.bigrams_min_boundary_be,
        bigrams_be_quantile=cfg.bigrams_be_quantile,
        bigrams_normalize_be=cfg.bigrams_normalize_be,
        bigrams_count_floor=cfg.bigrams_count_floor,
        bigram_iterations=cfg.bigram_iterations,
    )

    if cfg.dynamic_topk_floor > 0:
        keywords = _apply_dynamic_topk(
            keywords, floor=cfg.dynamic_topk_floor, min_k=cfg.dynamic_topk_min
        )

    # Save keywords
    keywords_path = curriculum_dir / "topic_keywords.json"
    keywords_json = {}
    for topic_idx, kws in keywords.items():
        if kws and isinstance(kws[0], tuple):
            keywords_json[topic_idx] = [[kw, float(score)] for kw, score in kws]
        else:
            keywords_json[topic_idx] = kws
    
    with open(keywords_path, "w") as f:
        json.dump(keywords_json, f, indent=2)
    LOGGER.info(f"Saved keywords to {keywords_path}")
    
    # Print summary
    LOGGER.info("")
    LOGGER.info("Topic Keywords:")
    outputs = load_curriculum_outputs(curriculum_dir)
    topic_work_ids = outputs.topic_work_ids or {}
    for topic_idx in sorted(keywords.keys()):
        topic_kws = keywords.get(topic_idx, [])
        if topic_kws and isinstance(topic_kws[0], tuple):
            kw_str = ", ".join([kw for kw, _ in topic_kws[:10]])
        else:
            kw_str = ", ".join([str(kw) for kw in topic_kws[:10]])
        size = len(topic_work_ids.get(str(topic_idx), []))
        LOGGER.info(f"  Topic {topic_idx} ({size} works): {kw_str}")
    
    return keywords


def run_plotting(
    cfg: PlotConfig,
    curriculum_dir: Path,
    embeddings: np.ndarray,
    topic_keywords: Optional[Dict[int, List[tuple[str, float]]]] = None,
) -> None:
    """Generate curriculum visualizations.
    
    Args:
        cfg: Plot configuration.
        curriculum_dir: Path to curriculum outputs directory.
        embeddings: Embeddings array (may be filtered to clustering subset).
        topic_keywords: Optional topic keywords for labels.
    """
    if not cfg.enabled:
        LOGGER.info("Plotting disabled, skipping...")
        return
    
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("CURRICULUM VISUALIZATION")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Config:\n{cfg}")
    
    # Convert figsize list to tuple for matplotlib
    figsize: tuple[int, int] = (cfg.figsize[0], cfg.figsize[1])
    
    # Load citation graph if requested
    citation_graph = None
    if cfg.umap.use_citation_graph:
        citation_graph_path = curriculum_dir / "citation_graph.npz"
        if citation_graph_path.exists():
            citation_graph = sp.load_npz(citation_graph_path)
            LOGGER.info(f"Loaded citation graph: {citation_graph.shape}")
        else:
            LOGGER.warning(f"Citation graph not found at {citation_graph_path}, ignoring use_citation_graph")
    
    if cfg.with_background:
        output_path = curriculum_dir / "curriculum_plot_with_background.png"
    else:
        output_path = curriculum_dir / "curriculum_plot.png"

    plot_curriculum(
        curriculum_dir=curriculum_dir,
        embeddings=embeddings,
        output_path=output_path,
        topic_keywords=topic_keywords,  # type: ignore[arg-type]
        figsize=figsize,
        dpi=cfg.dpi,
        method=cfg.method,
        umap_config=cfg.umap,
        lpp_config=cfg.lpp,
        citation_graph=citation_graph,
        background=cfg.with_background,
        max_points_per_topic=cfg.max_points_per_topic,
    )


@hydra.main(version_base=None, config_path=None, config_name="curriculum")
def main(cfg: CurriculumConfig) -> None:
    """Main entry point for the curriculum building pipeline.
    
    Runs curriculum building, keyword extraction, and visualization
    according to configuration.
    """
    LOGGER.info("=" * 60)
    LOGGER.info("CURRICULUM BUILDING PIPELINE")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Target work IDs: {cfg.target_work_ids}")
    LOGGER.info(f"Desired topics: {cfg.desired_n_topics}")
    LOGGER.info(f"Rank before clustering: {cfg.rank_before_clustering} (top_k={cfg.ranking_top_k})")
    LOGGER.info(f"Linkage method: {cfg.linkage_method}")
    LOGGER.info(f"Keywords enabled: {cfg.keywords.enabled}")
    LOGGER.info(f"Plotting enabled: {cfg.plot.enabled}")
    
    # Resolve output_dir: if left as default/empty, derive from overrides
    auto_name = cfg.output_dir in (None, "")
    if auto_name:
        output_dir_name = derive_output_dir_name(cfg)
    else:
        output_dir_name = str(cfg.output_dir)
    output_dir = get_curriculum_output_dir(output_dir_name, auto_derived=auto_name)
    
    LOGGER.info(f"Output directory: {output_dir}")

    # Reconfigure logging to write into the per-run directory
    setup_logging(
        log_name="pipeline",
        log_level="DEBUG",
        include_time=False,
        log_path=output_dir / "pipeline.log",
        explicit_args=None,
    )
    
    # Build curriculum (pass resolved output_dir to avoid re-creation)
    cloud, embeddings_for_clustering, filtered_indices = run_curriculum_building(cfg, output_dir=output_dir)
    
    # Keyword extraction (if enabled)
    topic_keywords = None
    if cfg.keywords.enabled:
        topic_keywords = run_keyword_extraction(
            cfg.keywords,
            output_dir,
            cloud,
            filtered_indices,
        )
    
    # Generate summary markdown
    generate_curriculum_summary(
        output_dir,
        cloud,
        filtered_indices,
        topic_keywords,
    )
    
    # Visualization (if enabled)
    if cfg.plot.enabled:
        run_plotting(
            cfg.plot,
            output_dir,
            embeddings_for_clustering,
            topic_keywords,
        )
    
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("PIPELINE COMPLETE")
    LOGGER.info("=" * 60)
    LOGGER.info(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
