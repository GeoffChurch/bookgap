"""Configuration for the standalone curriculum engine.

All knobs live in plain frozen-friendly dataclasses (no Hydra, no omegaconf —
those were bookgap couplings). The defaults reproduce the validated wiki-native
pipeline so the parity gate passes out of the box; the Lean / pure-graph path
flips ``embeddings_mode`` to ``SPECTRAL_FROM_GRAPH`` (or ``COMMUNITY_DETECT``)
and needs no embeddings.

The config is split along the engine's four axes — **rank**, **select**,
**cluster**, **order** — plus the optional output layers (keywords / quality /
judge) and the optional GDC diffusion densification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class RankMethod(str, Enum):
    """How relevance (PageRank) mass is seeded."""

    PAGE_RANK = "page_rank"
    """Forward personalized PageRank seeded at the targets (foundations accrue)."""

    BIDIRECTIONAL_PAGE_RANK = "bidirectional_page_rank"
    """Two-seed PageRank (targets forward x sources reverse), product-combined."""


class EmbeddingsMode(str, Enum):
    """Where node geometry (for clustering) comes from."""

    PROVIDED = "provided"
    """Use ``GraphSource.node_embeddings()`` directly (must be non-None)."""

    CALLABLE = "callable"
    """Call ``CurriculumConfig.embeddings_callable`` to produce embeddings."""

    SPECTRAL_FROM_GRAPH = "spectral_from_graph"
    """Laplacian Eigenmaps on the (hub-suppressed) graph — identity features.

    The Phase-0-bis pure-graph path: no semantic embedder. Derives a k-dim
    spectral embedding from the adjacency alone via sparse ``eigsh`` on the
    hub-suppressed normalized Laplacian.
    """

    COMMUNITY_DETECT = "community_detect"
    """Cluster the graph directly (spectral community detection); no embeddings.

    Sets cluster labels straight from the graph; skips the embedding +
    dendrogram path entirely.
    """


class OrderingMode(str, Enum):
    """How discovered topics are linearized (prerequisite -> frontier)."""

    FRONTIER_ADJACENCY = "frontier_adjacency"
    """Inter-topic dependency via absorbing PageRank, then in-degree - out-degree.

    The validated default: builds an ``(|T|, |T|)`` dependency matrix with
    ``compute_frontier_adjacency`` over the RAW forward-PR flow, then orders by
    (depended-on - depends-on) so foundations come first.
    """

    IN_OUT_DEGREE = "in_out_degree"
    """Alias of frontier_adjacency's final linearization (kept for clarity)."""

    MASS = "mass"
    """Order by total raw forward-PR mass per topic (cheap; no |T|x|T| solve).

    Reproduces ``wiki_curriculum_demo``'s default ``--order-by mass``.
    """


class SelectionMode(str, Enum):
    """How the top-N clustering pool is scored."""

    TOP_N = "top_n"
    """Top-N by raw forward-PR mass."""

    BACKGROUND_NORMALIZED = "background_normalized"
    """Top-N by ``p^a * log((p+eps)/(g+eps))`` (demotes globally-popular hubs)."""


class DiffusionKernel(str, Enum):
    HEAT = "heat"
    PPR = "ppr"


@dataclass
class PageRankConfig:
    """Power-iteration PageRank parameters."""

    alpha: float = 0.85
    """Damping factor (probability of following an edge vs teleporting)."""

    tol: float = 1e-9
    """Convergence tolerance (max abs change in the mass vector)."""

    max_iter: int = 200
    """Maximum power-iteration steps."""


@dataclass
class BackgroundNormConfig:
    """Background-normalized selection score knobs.

    Score = ``p^a * log((p+eps)/(g+eps))`` where ``p`` is the seeded forward
    PR and ``g`` is a uniform-teleport background PR on the same adjacency.
    """

    exponent: float = 0.5
    """``a``: 0 = raw log-lift, 0.5 = Fightin'-Words variance-norm (canonical),
    1.0 = pointwise KL."""

    alpha: Optional[float] = None
    """Damping for the background PR. ``None`` => match ``pagerank.alpha``."""

    eps: float = 1e-12
    """Laplace floor in ``log((p+eps)/(g+eps))``; guards p=0 nodes."""


@dataclass
class SpectralEmbeddingConfig:
    """Pure-graph spectral embedding (Laplacian Eigenmaps) knobs.

    Computes the bottom-``n_components`` eigenvectors of the hub-suppressed
    normalized Laplacian directly via sparse ``eigsh`` — never forms a dense
    ``X L X^T``. This is identity-feature link-LPP (see the package docstring
    on ``layers.lpp``).
    """

    n_components: int = 64
    """Number of spectral dimensions (the clustering subspace dim)."""

    hub_suppress: bool = True
    """Down-weight each edge into target ``j`` by ``1/log(indeg_j + e)`` before
    symmetrizing (the link-graph hub pathology fix)."""


@dataclass
class DiffusionConfig:
    """Optional Graph Diffusion Convolution (GDC) densification of the graph.

    Off by default. When ``enabled``, the adjacency is re-wired by a smooth
    multi-hop heat/PPR kernel, re-sparsified to top-``top_m``/row, renormalized,
    with the directed skeleton overlaid so forward PageRank keeps direction.
    """

    enabled: bool = False
    kernel: DiffusionKernel = DiffusionKernel.HEAT
    t: float = 8.0
    """Heat-kernel diffusion time (used when kernel=HEAT)."""
    ppr_alpha: float = 0.15
    """PPR restart probability (used when kernel=PPR)."""
    top_m: int = 32
    """Keep top-``top_m`` entries per row after diffusion."""
    series_terms: int = 12
    """Truncation order of the diffusion series."""
    cap_m: int = 64
    """Per-row cap on intermediate powers (keeps the series sparse)."""


@dataclass
class ClusteringConfig:
    """Clustering / dendrogram knobs (the embedding-rich path)."""

    enabled: bool = True
    """If False, every selected node forms one trivial cluster (minimal core)."""

    desired_n_topics: int = 12
    """Target number of frontier topics."""

    min_cluster_size: int = 1
    """Bisecting-KMeans: stop splitting clusters smaller than this."""

    weighted: bool = False
    """Weight the bisection by per-node relevance mass (anchors high-mass nodes)."""

    community_n_clusters: Optional[int] = None
    """For ``embeddings_mode=COMMUNITY_DETECT``: exact cluster count (defaults
    to ``desired_n_topics`` when None)."""


@dataclass
class KeywordsConfig:
    """Keyword-extraction layer knobs (log-odds + lattice phrase mining)."""

    enabled: bool = True
    top_k: int = 12
    samples_per_topic: int = 80
    bigrams: bool = True
    bigram_iterations: int = 6
    npmi_threshold: float = 0.4
    min_boundary_be: float = 0.5
    count_floor: int = 3
    max_phrases: int = 5000


@dataclass
class SelectionConfig:
    """Top-N selection knobs."""

    mode: SelectionMode = SelectionMode.TOP_N
    top_n: int = 4000
    """Number of nodes to keep for clustering, by selection score."""

    per_corpus: bool = False
    """If True, z-normalize the selection score WITHIN each corpus (node_type)
    before the top-N cut, so a smaller corpus is not starved (joint-graph)."""

    background: BackgroundNormConfig = field(default_factory=BackgroundNormConfig)


@dataclass
class JudgeConfig:
    """LLM-as-judge layer knobs (local ollama)."""

    enabled: bool = False
    model: str = "gpt-oss:20b"
    repeats: int = 1
    temperature: float = 0.0
    num_ctx: int = 4096
    num_predict: int = 1536
    n_order_pairs: int = 10
    seed: int = 0
    rawfail_path: Optional[str] = None
    """If set, raw LLM responses that fail to parse are appended here (jsonl)."""


@dataclass
class CurriculumConfig:
    """Top-level config for :func:`curriculum.build.build_curriculum`.

    Defaults reproduce the wiki-native pipeline (mass-order to match the
    committed ``docs/wikipedia-curriculum.md``; flip ``ordering`` to
    ``FRONTIER_ADJACENCY`` for the directional DAG order). The Lean / pure-graph
    path sets ``embeddings_mode = SPECTRAL_FROM_GRAPH`` (or ``COMMUNITY_DETECT``).
    """

    # --- rank ---
    method: RankMethod = RankMethod.PAGE_RANK
    pagerank: PageRankConfig = field(default_factory=PageRankConfig)

    # --- select ---
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    # --- node geometry / cluster ---
    embeddings_mode: EmbeddingsMode = EmbeddingsMode.PROVIDED
    spectral: SpectralEmbeddingConfig = field(default_factory=SpectralEmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)

    # --- order ---
    ordering: OrderingMode = OrderingMode.MASS

    # --- optional layers ---
    keywords: KeywordsConfig = field(default_factory=KeywordsConfig)
    quality_enabled: bool = False
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    # --- optional graph rewiring ---
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)

    # --- pluggable embedder (embeddings_mode=CALLABLE) ---
    embeddings_callable: Optional[object] = None
    """Callable ``(GraphSource) -> (n_nodes, dim) float array`` used when
    ``embeddings_mode == CALLABLE``. Not part of (de)serialization."""

    # --- misc ---
    random_state: int = 42

    def background_alpha(self) -> float:
        """Resolve the background-PR damping (defaults to the foreground alpha)."""
        bg = self.selection.background.alpha
        return float(bg) if bg is not None else float(self.pagerank.alpha)
