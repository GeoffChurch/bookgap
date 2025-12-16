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
    DocWeighting,
    KeywordMethod,
    TopicWeighting,
    extract_topic_keywords,
    filter_stopwords,
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
) -> Dict[int, List[tuple[str, float]]]:
    """Generate keyword summaries for curriculum topics.
    
    Args:
        curriculum_dir: Path to directory containing curriculum outputs.
        texts: List of combined title+abstract strings (ordered to match cloud lists).
        samples_per_topic: Number of works to sample from each topic.
        top_k_keywords: Number of top keywords to extract per topic.
        method: Keyword extraction method ('tfidf' or 'log_odds').
        
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
    
    # Sample works and extract text for each topic
    LOGGER.info(f"Sampling up to {samples_per_topic} works per topic using PageRank relevance...")
    # Keep per-document tokens so keyword extraction can choose weighting.
    # Shape: topics -> docs -> tokens
    topic_doc_tokens: List[List[List[str]]] = []
    
    for topic_idx in tqdm(range(n_topics), desc="Sampling topics"):
        # Get all works in this topic using point_to_cluster mapping
        topic_mask = outputs.point_to_cluster == topic_idx
        topic_work_indices = np.where(topic_mask)[0]
        
        # Select top works by PageRank relevance
        sampled_indices = sample_by_relevance(
            topic_work_indices,
            outputs.point_masses,
            samples_per_topic,
        )
        
        # Extract and tokenize text from sampled works (texts list already contains
        # combined title+abstract strings at the same ordering as the cloud lists)
        doc_tokens: List[List[str]] = []
        for work_idx in sampled_indices:
            if work_idx < n_works:
                text = texts[work_idx]
                tokens = tokenize(text)
                tokens = filter_stopwords(tokens)
                if tokens:
                    doc_tokens.append(tokens)
        
        topic_doc_tokens.append(doc_tokens)
        LOGGER.debug(
            f"Topic {topic_idx}: sampled {len(sampled_indices)} works, "
            f"{sum(len(t) for t in doc_tokens)} tokens"
        )
    
    # Extract keywords
    LOGGER.info(f"Extracting keywords using {method} method...")
    topic_keywords = extract_topic_keywords(
        topic_doc_tokens,
        method=method,
        top_k=top_k_keywords,
        doc_weighting=doc_weighting,
        topic_weighting=topic_weighting,
    )
    
    # Convert to dictionary
    result = {i: keywords for i, keywords in enumerate(topic_keywords)}
    
    return result

