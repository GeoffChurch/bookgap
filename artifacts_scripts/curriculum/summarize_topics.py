#!/usr/bin/env python3
"""Generate topic keyword summaries for curriculum topics.

This module provides functions to generate "bag of keywords" summaries
for each frontier topic using PageRank-based relevance scoring and
statistical keyword extraction.

The relevance-based sampling selects the most relevant works from each
topic according to the original PageRank scores (not inter-topic PageRank).

Use via the main.py unified entry point or import functions directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from bookgap.util import LOGGER

from .keyword_extraction import (
    BackgroundMode,
    DocWeighting,
    KeywordMethod,
    TopicWeighting,
    extract_topic_keywords,
    filter_stopwords,
    mine_phrases_lattice,
    parse_with_phrases,
    tokenize,
)
from .sampling import sample_by_relevance


@dataclass(frozen=True)
class CurriculumOutputs:
    """Curriculum building outputs loaded from disk.
    
    Attributes:
        frontier_nodes: Indices of frontier topic nodes in dendrogram.
        node_masses: Masses propagated up dendrogram tree.
        linked: Scipy linkage matrix from hierarchical clustering.
        point_to_cluster: Mapping from leaf indices to frontier clusters.
        leaf_cluster_assignments: Optional mapping from points to dendrogram leaves.
        point_masses: Optional PageRank scores for original personalization.
        leaf_masses: Optional aggregated masses per leaf cluster.
        metadata: Optional metadata dictionary from curriculum building.
        topic_work_ids: Optional mapping from topic indices to work IDs.
    """
    frontier_nodes: NDArray[np.intp]
    node_masses: NDArray[np.floating]
    linked: NDArray[np.floating]
    point_to_cluster: NDArray[np.intp]
    leaf_cluster_assignments: Optional[NDArray[np.intp]] = None
    point_masses: Optional[NDArray[np.floating]] = None
    leaf_masses: Optional[NDArray[np.floating]] = None
    metadata: Optional[Dict[str, Any]] = None
    topic_work_ids: Optional[Dict[str, List[Any]]] = None


def load_curriculum_outputs(curriculum_dir: Path) -> CurriculumOutputs:
    """Load curriculum outputs from a directory.
    
    Args:
        curriculum_dir: Path to directory containing curriculum outputs.
        
    Returns:
        CurriculumOutputs dataclass with loaded arrays and metadata.
        
    Raises:
        FileNotFoundError: If required files are missing.
    """
    # Load numpy arrays - handle both old and new naming conventions
    required_arrays = ["frontier_nodes", "node_masses", "linked"]
    loaded_arrays: Dict[str, NDArray] = {}
    for name in required_arrays:
        path = curriculum_dir / f"{name}.npy"
        if path.exists():
            loaded_arrays[name] = np.load(path)
        else:
            raise FileNotFoundError(f"Required file not found: {path}")
    
    # Handle point_to_cluster / leaf_to_cluster (new vs old naming)
    point_to_cluster_path = curriculum_dir / "point_to_cluster.npy"
    leaf_to_cluster_path = curriculum_dir / "leaf_to_cluster.npy"
    
    if point_to_cluster_path.exists():
        point_to_cluster = np.load(point_to_cluster_path)
    elif leaf_to_cluster_path.exists():
        point_to_cluster = np.load(leaf_to_cluster_path)
    else:
        raise FileNotFoundError(
            f"Neither point_to_cluster.npy nor leaf_to_cluster.npy found in {curriculum_dir}"
        )
    
    # Load optional arrays
    optional_arrays: Dict[str, Optional[NDArray]] = {
        "leaf_cluster_assignments": None,
        "point_masses": None,
        "leaf_masses": None,
    }
    for name in optional_arrays:
        path = curriculum_dir / f"{name}.npy"
        if path.exists():
            optional_arrays[name] = np.load(path)
    
    # Load metadata
    metadata = None
    metadata_path = curriculum_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        raise FileNotFoundError(f"Required file not found: {metadata_path}")
    
    # Load topic work IDs
    topic_work_ids = None
    topic_ids_path = curriculum_dir / "topic_work_ids.json"
    if topic_ids_path.exists():
        with open(topic_ids_path) as f:
            topic_work_ids = json.load(f)
    
    return CurriculumOutputs(
        frontier_nodes=loaded_arrays["frontier_nodes"],
        node_masses=loaded_arrays["node_masses"],
        linked=loaded_arrays["linked"],
        point_to_cluster=point_to_cluster,
        leaf_cluster_assignments=optional_arrays["leaf_cluster_assignments"],
        point_masses=optional_arrays["point_masses"],
        leaf_masses=optional_arrays["leaf_masses"],
        metadata=metadata,
        topic_work_ids=topic_work_ids,
    )


def summarize_topics(
    curriculum_dir: Path,
    texts: List[str],
    samples_per_topic: int = 50,
    top_k_keywords: int = 20,
    method: KeywordMethod = KeywordMethod.LOG_ODDS,
    doc_weighting: DocWeighting = DocWeighting.EQUAL,
    topic_weighting: TopicWeighting = TopicWeighting.EQUAL,
    background_mode: BackgroundMode = BackgroundMode.MARGINAL,
    weight_by_freq: bool = True,
    bigrams_enabled: bool = True,
    bigrams_npmi_threshold: float = 0.4,
    bigrams_max: int = 5000,
    bigrams_min_boundary_be: float = 0.5,
    bigrams_be_quantile: float = 0.0,
    bigrams_normalize_be: bool = False,
    bigrams_count_floor: int = 3,
    bigram_iterations: int = 6,
    embeddings: Optional["np.ndarray"] = None,
    soft_topics: bool = False,
    soft_mode: str = "embedding",
    soft_perplexity: float = 1.5,
    soft_alpha: float = 0.5,
    soft_floor: float = 0.05,
    citation_adj=None,
) -> Dict[int, List[tuple[str, float]]]:
    """Generate keyword summaries for curriculum topics.

    Args:
        curriculum_dir: Path to directory containing curriculum outputs.
        texts: List of combined title+abstract strings (ordered to match cloud lists).
        samples_per_topic: Number of works to sample from each topic.
        top_k_keywords: Number of top keywords to extract per topic.
        method: Keyword extraction method ('tfidf' or 'log_odds').
        doc_weighting: How to weight documents. If RELEVANCE, uses point_masses.
        topic_weighting: How to weight topics in background construction
            (only meaningful when background_mode=SUBTRACTIVE).
        background_mode: For log-odds, MARGINAL (unary, default) or
            SUBTRACTIVE (legacy "vs. all other topics").
        weight_by_freq: For log-odds, multiply z-scores by topic-internal
            count to rank by per-word MI contribution. Default True.
        bigrams_enabled: Mine adjacent-token phrases from the sampled
            corpus using ``mine_phrases_lattice`` (combined NPMI · count
            · boundary-entropy criterion), apply them via longest-match
            ``parse_with_phrases`` before keyword extraction. Surfaces
            phrases like ``bell_theorem``, ``single_photon``,
            ``photon_counting`` while rejecting boilerplate aggregates
            (``articles_citing``, ``advertisement_return``) that NPMI
            alone admits.
        bigrams_npmi_threshold: Minimum NPMI for bigram candidates.
        bigrams_max: Cap on phrases kept *per iteration*; the cumulative
            phrase vocabulary across iterations can be larger.
        bigrams_min_boundary_be: Absolute boundary-entropy floor in bits
            (default 0.5). Stable across the iterative loop's BE
            distribution shifts.
        bigrams_be_quantile: Optional data-driven BE threshold (quantile
            of the candidate BE distribution). Combined with
            ``bigrams_min_boundary_be`` via ``max``: a candidate must
            clear both. Default 0.0 disables the quantile filter.
        bigrams_normalize_be: If True, score uses BE / log2(count+1)
            (count-normalised, intrinsically iteration-stable) rather
            than absolute BE. Threshold filters still apply to the
            un-normalised BE.
        bigrams_count_floor: Minimum count per candidate.
        bigram_iterations: Maximum iterative phrase-mining passes. The
            loop self-terminates when a pass adds no new phrases. Default
            6 is a generous cap — typical corpora exit after 2–3 passes.
            Each iteration re-parses the *raw* token stream against the
            cumulative phrase vocabulary using longest-match, so already-
            merged phrases consume their constituent positions and don't
            appear as separate adjacencies in subsequent passes (the
            semtok-style "exclusive entropy" effect, achieved by
            re-parsing rather than explicit count subtraction).

    Returns:
        Dictionary mapping topic index to list of (keyword, score) tuples.
    """
    # Load curriculum outputs
    LOGGER.info(f"Loading curriculum outputs from {curriculum_dir}...")
    outputs = load_curriculum_outputs(curriculum_dir)
    
    if outputs.point_masses is None:
        raise ValueError("point_masses not found in curriculum outputs")
    
    n_topics = len(outputs.frontier_nodes)
    n_works = len(texts)

    LOGGER.info(f"Found {n_topics} frontier topics, {n_works} works")

    # Optional fuzzy/soft topic membership: a work contributes to every topic
    # with responsibility >= soft_floor (weighted mass*r), not just its hard topic.
    # NOTE: opt-in / ~5x slower than hard assignment — the soft pools are larger
    # (a work enters multiple topics' phrase-mining pools below), so the iterative
    # phrase mining does much more work. Probably a quality win (dissolves degenerate
    # chapters); enable deliberately. See cfg.keywords.soft_topics docstring.
    soft_R = None
    if soft_topics:
        labels = np.asarray(outputs.point_to_cluster)
        if soft_mode == "citation":
            # Absorbing-walk arm: diffuse the citation-coherent hard labels through
            # the citation graph (label propagation). Citation-coherent membership.
            if citation_adj is None:
                raise ValueError("soft_topics soft_mode='citation' requires citation_adj")
            from artifacts_scripts.curriculum.gap_fuzzy_topics import citation_label_propagation
            soft_R = citation_label_propagation(citation_adj, labels, n_topics, alpha=soft_alpha)
            LOGGER.info(
                "Soft topics (citation label-propagation): alpha=%.2f, floor=%.2f — works "
                "contribute to all topics with r>=floor, weighted mass*r", soft_alpha, soft_floor,
            )
        else:
            if embeddings is None:
                raise ValueError("soft_topics soft_mode='embedding' requires embeddings")
            from artifacts_scripts.curriculum.gap_fuzzy_topics import soft_responsibilities, _unit
            masses = np.asarray(outputs.point_masses, dtype=np.float64)
            E = _unit(np.asarray(embeddings, dtype=np.float32))
            C = np.zeros((n_topics, E.shape[1]), dtype=np.float32)
            for k in range(n_topics):
                m = np.where(labels == k)[0]
                if len(m) == 0:
                    continue
                w = masses[m]
                C[k] = np.average(E[m], axis=0, weights=w if w.sum() > 0 else None)
            C = _unit(C)
            soft_R, tau = soft_responsibilities(E, C, soft_perplexity)
            LOGGER.info(
                "Soft topics (embedding softmax): tau=%.4f (target perplexity %.2f), floor=%.2f "
                "— works contribute to all topics with r>=floor, weighted mass*r",
                tau, soft_perplexity, soft_floor,
            )
    
    # Sample works and extract text for each topic
    LOGGER.info(f"Sampling up to {samples_per_topic} works per topic using PageRank relevance...")
    # Keep per-document tokens so keyword extraction can choose weighting.
    # Shape: topics -> docs -> tokens
    topic_doc_tokens: List[List[List[str]]] = []
    # Shape: topics -> docs -> weight (for RELEVANCE mode)
    topic_doc_weights: Optional[List[List[float]]] = (
        [] if doc_weighting == DocWeighting.RELEVANCE else None
    )
    
    # Pass 1: sample + tokenize (no stopword filter yet — bigram mining needs
    # raw token sequences to avoid merging across removed stopwords). Track
    # per-(topic, doc) raw tokens and relevance weights for the second pass.
    topic_raw_tokens: List[List[List[str]]] = []
    topic_doc_relevances: List[List[float]] = []
    for topic_idx in tqdm(range(n_topics), desc="Sampling topics"):
        if soft_R is not None:
            resp = soft_R[:, topic_idx]
            topic_work_indices = np.where(resp >= soft_floor)[0]
            sampling_weights = np.asarray(outputs.point_masses, dtype=np.float64) * resp
        else:
            topic_work_indices = np.where(outputs.point_to_cluster == topic_idx)[0]
            sampling_weights = outputs.point_masses
        sampled_indices = sample_by_relevance(
            topic_work_indices,
            sampling_weights,
            samples_per_topic,
        )

        raw_tokens_per_doc: List[List[str]] = []
        relevances_per_doc: List[float] = []
        for work_idx in sampled_indices:
            if work_idx < n_works:
                tokens = tokenize(texts[work_idx])
                if tokens:
                    raw_tokens_per_doc.append(tokens)
                    w = float(outputs.point_masses[work_idx])
                    if soft_R is not None:
                        w *= float(soft_R[work_idx, topic_idx])
                    relevances_per_doc.append(w)
        topic_raw_tokens.append(raw_tokens_per_doc)
        topic_doc_relevances.append(relevances_per_doc)

    # Iterative lattice phrase mining (semtok-style exclusive entropy via
    # re-parse): each pass scores adjacent-token bigrams over the *current
    # parse* of the raw corpus, adds new ones to the cumulative phrase
    # vocabulary, and re-parses the raw corpus from scratch with
    # longest-match against the updated vocab. Already-merged phrases
    # consume their constituent positions on every pass, so subsequent
    # boundary-entropy and NPMI computations naturally exclude counts
    # that belong to longer phrases.
    if bigrams_enabled and bigram_iterations >= 1:
        flat_raw: List[List[str]] = [
            doc for topic_docs in topic_raw_tokens for doc in topic_docs
        ]
        phrase_vocab: set = set()
        parsed = flat_raw  # iter 0 parse = raw
        for it in range(bigram_iterations):
            new_phrases = mine_phrases_lattice(
                parsed,
                npmi_threshold=bigrams_npmi_threshold,
                min_boundary_be=bigrams_min_boundary_be,
                be_quantile=bigrams_be_quantile,
                normalize_be=bigrams_normalize_be,
                count_floor=bigrams_count_floor,
                max_phrases=bigrams_max,
            )
            new_phrases -= phrase_vocab  # only newly added this iteration
            if not new_phrases:
                LOGGER.info(
                    f"Phrase mining iter {it + 1}: no new phrases "
                    f"above threshold; stopping."
                )
                break
            phrase_vocab |= new_phrases
            LOGGER.info(
                f"Phrase mining iter {it + 1}: "
                f"+{len(new_phrases)} new phrases (vocab={len(phrase_vocab)}; "
                f"npmi>={bigrams_npmi_threshold}, "
                f"be>={bigrams_min_boundary_be} bits / q={bigrams_be_quantile}, "
                f"count>={bigrams_count_floor})"
            )
            # Re-parse raw with cumulative phrase vocab for next iteration's
            # mining (and the final downstream parse).
            parsed = [parse_with_phrases(doc, phrase_vocab) for doc in flat_raw]

        # Distribute final parse back into per-topic structure.
        merged_per_topic: List[List[List[str]]] = []
        offset = 0
        for topic_docs in topic_raw_tokens:
            n = len(topic_docs)
            merged_per_topic.append(parsed[offset : offset + n])
            offset += n
        topic_raw_tokens = merged_per_topic

    # Final pass: filter stopwords (bigrams use ``_`` join char so they survive),
    # collect per-topic structures for keyword extraction.
    for topic_idx in range(n_topics):
        doc_tokens: List[List[str]] = []
        doc_weights: List[float] = []
        for tokens, relevance in zip(
            topic_raw_tokens[topic_idx], topic_doc_relevances[topic_idx],
        ):
            tokens = filter_stopwords(tokens)
            if tokens:
                doc_tokens.append(tokens)
                if topic_doc_weights is not None:
                    doc_weights.append(relevance)
        topic_doc_tokens.append(doc_tokens)
        if topic_doc_weights is not None:
            topic_doc_weights.append(doc_weights)
    
    # Extract keywords
    LOGGER.info(f"Extracting keywords using {method} method...")
    topic_keywords = extract_topic_keywords(
        topic_doc_tokens,
        method=method,
        top_k=top_k_keywords,
        doc_weighting=doc_weighting,
        topic_weighting=topic_weighting,
        topic_doc_weights=topic_doc_weights,
        background_mode=background_mode,
        weight_by_freq=weight_by_freq,
    )
    
    # Convert to dictionary
    result = {i: keywords for i, keywords in enumerate(topic_keywords)}
    
    return result

