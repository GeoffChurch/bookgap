"""Configuration for curriculum building."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from bookgap.configs.config import ExperimentConfig
from bookgap.configs.gap.gap_config import GapConfig
from bookgap.configs.llm.llm_config import LLMConfig
from bookgap.configs.batch.batch_config import BatchConfig
from bookgap.configs.output.output_config import OutputConfig
from bookgap.configs.cloud.cloud_config import CloudConfig
from bookgap.util import ARTIFACTS_DIR

from .dimensionality_reduction import DimensionalityReductionMethod
from .keyword_extraction import DocWeighting, KeywordMethod, TopicWeighting


def get_curriculum_output_dir(output_dir: str) -> Path:
    """Get the output directory for curriculum artifacts.
    
    Args:
        output_dir: Subdirectory name under artifacts/curriculum.
        
    Returns:
        Path to the output directory (created if it doesn't exist).
    """
    path = Path(ARTIFACTS_DIR) / "curriculum" / output_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class KeywordConfig:
    """Configuration for keyword extraction."""
    
    enabled: bool = True
    """Whether to run keyword extraction after curriculum building."""
    
    method: KeywordMethod = KeywordMethod.LOG_ODDS
    """Keyword extraction method."""
    
    top_k: int = 100
    """Number of top keywords per topic."""
    
    samples_per_topic: int = 5000
    """Number of works to sample per topic for keyword extraction."""

    doc_weighting: DocWeighting = DocWeighting.EQUAL
    """How to weight documents within each topic when building keyword statistics.

    - DocWeighting.EQUAL: each sampled work contributes equal mass (document-length invariant)
    - DocWeighting.TOKEN: longer texts contribute proportionally more (bag-of-tokens)
    """

    topic_weighting: TopicWeighting = TopicWeighting.EQUAL
    """How to weight topics when constructing the log-odds background distribution.

    - TopicWeighting.POOLED: background is pooled across all other topics (larger topics contribute more)
    - TopicWeighting.EQUAL: each topic contributes equal total mass to the background

    Note: currently applies to keywords.method="log_odds".
    """


@dataclass
class UMAPConfig:
    """UMAP-specific parameters."""
    
    n_neighbors: int = 15
    """Number of neighbors to consider in k-NN graph construction."""
    
    min_dist: float = 0.1
    """Minimum distance between points in 2D projection."""
    
    use_citation_graph: bool = True
    """If True, use citation graph as connectivity structure instead of embeddings.
    
    When enabled, UMAP will use the citation graph to guide the projection,
    keeping papers that cite each other close together. This requires that
    citation_graph.npz exists in the curriculum output directory.
    """


@dataclass
class PCAConfig:
    """PCA-specific parameters."""
    
    pass  # Currently no PCA-specific params, but extensible


@dataclass
class LDAConfig:
    """LDA-specific parameters."""
    
    pass  # Currently no LDA-specific params, but extensible


@dataclass
class LPPConfig:
    """LPP-specific parameters."""

    center: bool = True
    """Whether to center embeddings before solving the GEP (default: True)."""

    n_components: int = 2
    """Number of projection dimensions to return (default: 2)."""


@dataclass
class PlotConfig:
    """Configuration for curriculum visualization."""
    
    enabled: bool = True
    """Whether to generate plots after curriculum building."""
    
    with_background: bool = True
    """If True, plot individual points behind topic centroids."""
    
    figsize: List[int] = field(default_factory=lambda: [12, 10])
    """Figure size in inches [width, height]."""
    
    dpi: int = 300
    """Resolution for saved figures."""
    
    max_points_per_topic: int = 300
    """Max points to plot per topic (for background mode)."""
    
    method: DimensionalityReductionMethod = DimensionalityReductionMethod.UMAP
    """Dimensionality reduction method for 2D projection.
    
    Options: DimensionalityReductionMethod.PCA, LDA, or UMAP
    
    - PCA: Fast, unsupervised, linear
    - LDA: Supervised, maximizes topic separation
    - UMAP: Supervised, preserves local and global structure (requires umap-learn package)
    """
    
    # Nested configs for each method
    pca: PCAConfig = field(default_factory=PCAConfig)
    """PCA-specific configuration."""
    
    lda: LDAConfig = field(default_factory=LDAConfig)
    """LDA-specific configuration."""
    
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    """UMAP-specific configuration."""

    lpp: LPPConfig = field(default_factory=LPPConfig)
    """LPP-specific configuration."""


@dataclass
class CurriculumConfig:
    """Configuration for curriculum building.
    
    Builds a curriculum for a target topic by mining prerequisite relationships
    using personalized PageRank on the citation graph.
    """
    
    target_work_ids: List[str] = field(default_factory=list)
    """List of OpenAlex work IDs defining the target topic (seed set for PageRank)."""
    
    desired_n_topics: int = 10
    """Desired number of frontier topics in the curriculum."""
    
    pagerank_alpha: float = 0.90
    """Damping factor for PageRank (probability of following a citation vs teleporting)."""
    
    pagerank_tol: float = 1e-10
    """Convergence tolerance for PageRank iteration."""
    
    pagerank_max_iter: int = 500
    """Maximum number of PageRank iterations."""
    
    linkage_method: str = "bisecting_kmeans"
    """Linkage method for hierarchical clustering.
    
    Options:
    - 'ward', 'average', 'complete', 'single': scipy agglomerative clustering
      (requires O(n²) memory, suitable for n < 100k)
    - 'bisecting_kmeans': recursive bisecting K-Means
      (O(n) memory, suitable for n > 100k)
    """
    
    min_cluster_size: int = 1
    """For bisecting_kmeans: stop splitting clusters smaller than this.
    
    Larger values make clustering faster but coarser.
    - 1: Split to individual points (slowest, finest)
    - 100: Stop at ~100-point clusters (faster)
    - 1000: Stop at ~1000-point clusters (much faster for large datasets)
    """
    
    max_cluster_depth: Optional[int] = None
    """For bisecting_kmeans: maximum tree depth. None means no limit.
    
    Useful for very large datasets to limit recursion depth.
    - None: No limit (default)
    - 20: At most 2^20 = 1M leaf clusters
    - 15: At most 2^15 = 32k leaf clusters
    """
    
    ppr_first: bool = False
    """If True, run PPR first and cluster only the top-k relevant works.
    
    This is more efficient for large datasets and produces more coherent topics
    since only prerequisite-relevant works are clustered.
    """
    
    ppr_top_k: int = 500000
    """Number of top works by PPR mass to include in clustering when ppr_first=True.
    
    With 100k works, Ward's linkage is tractable (~8GB memory).
    """
    
    # Sub-configs for keyword extraction and plotting
    keywords: KeywordConfig = field(default_factory=KeywordConfig)
    """Configuration for keyword extraction."""
    
    plot: PlotConfig = field(default_factory=PlotConfig)
    """Configuration for curriculum visualization."""
    
    # Provide a safe default experiment composed of default sub-configs
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(
            gap=GapConfig(),
            llm=LLMConfig(),
            batch=BatchConfig(),
            output=OutputConfig(),
            cloud=CloudConfig(),
        )
    )
    """Experiment configuration (cloud settings, etc.)."""
    
    output_dir: str = "default"
