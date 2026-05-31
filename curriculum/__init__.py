"""Standalone, input-agnostic curriculum engine.

Turns a ``(graph, [optional embeddings], seeds, node-text)`` quadruple into an
ordered, clustered, keyworded curriculum: **seeded / background-normalized
PageRank** (relevance) -> **top-N selection** -> **clustering** (topics) ->
**frontier-adjacency directional ordering** (prerequisite -> frontier).

Fully decoupled from bookgap: the core (``curriculum.core``) imports only
stdlib + numpy/scipy/scikit-learn; the only optional third-party dependency is
``ollama`` (for the LLM-judge layer), lazily imported. Build a curriculum from a
graph alone (no embeddings) via ``embeddings_mode='spectral_from_graph'`` or
``'community_detect'`` — the Lean / pure-graph path.

Public surface::

    from curriculum import build_curriculum, CurriculumConfig
    from curriculum.graph_source import GraphSource, InMemoryGraphSource
"""

from __future__ import annotations

from curriculum.build import Curriculum, build_curriculum
from curriculum.config import (
    BackgroundNormConfig,
    ClusteringConfig,
    CurriculumConfig,
    DiffusionConfig,
    EmbeddingsMode,
    JudgeConfig,
    KeywordsConfig,
    OrderingMode,
    PageRankConfig,
    RankMethod,
    SelectionConfig,
    SelectionMode,
    SpectralEmbeddingConfig,
)
from curriculum.graph_source import GraphSource, InMemoryGraphSource

__all__ = [
    "build_curriculum",
    "Curriculum",
    "CurriculumConfig",
    "GraphSource",
    "InMemoryGraphSource",
    "RankMethod",
    "EmbeddingsMode",
    "OrderingMode",
    "SelectionMode",
    "PageRankConfig",
    "SelectionConfig",
    "BackgroundNormConfig",
    "ClusteringConfig",
    "KeywordsConfig",
    "JudgeConfig",
    "DiffusionConfig",
    "SpectralEmbeddingConfig",
]
