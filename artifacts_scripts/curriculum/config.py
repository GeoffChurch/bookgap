"""Configuration for curriculum building."""

from dataclasses import dataclass, field, fields, is_dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Mapping

from omegaconf import OmegaConf, DictConfig

from bookgap.configs.config import ExperimentConfig
from bookgap.configs.gap.gap_config import GapConfig
from bookgap.configs.llm.llm_config import LLMConfig
from bookgap.configs.batch.batch_config import BatchConfig
from bookgap.configs.output.output_config import OutputConfig
from bookgap.configs.cloud.cloud_config import CloudConfig
from bookgap.util import ARTIFACTS_DIR, create_unicode_safe_filename

from .dimensionality_reduction import DimensionalityReductionMethod
from .keyword_extraction import BackgroundMode, DocWeighting, KeywordMethod, TopicWeighting

@dataclass
class KeywordConfig:
    """Configuration for keyword extraction."""

    enabled: bool = True
    """Whether to run keyword extraction after curriculum building."""

    method: KeywordMethod = KeywordMethod.LOG_ODDS
    """Keyword extraction method."""

    top_k: int = 20
    """Number of top keywords per topic."""

    samples_per_topic: int = 5000
    """Number of works to sample per topic for keyword extraction."""

    doc_weighting: DocWeighting = DocWeighting.RELEVANCE
    """How to weight documents within each topic when building keyword statistics.

    - DocWeighting.RELEVANCE: each sampled work contributes mass proportional to its
      CircuitRank/PageRank relevance score. High-relevance papers dominate keywords.
    - DocWeighting.EQUAL: each sampled work contributes equal mass (document-length invariant)
    - DocWeighting.TOKEN: longer texts contribute proportionally more (bag-of-tokens)
    """

    topic_weighting: TopicWeighting = TopicWeighting.EQUAL
    """How to weight topics when constructing the log-odds background distribution.

    - TopicWeighting.POOLED: larger topics contribute more
    - TopicWeighting.EQUAL: each topic contributes equal total mass to the background

    Note: currently applies to keywords.method="log_odds".
    """

    background_mode: BackgroundMode = BackgroundMode.MARGINAL
    """For log-odds: how to construct the comparison background.

    - BackgroundMode.MARGINAL (default): unary test against the curriculum-
      wide marginal P(w). No "vs. what corpus" hyperparameter; domain words
      common across topics demote themselves automatically. Pairs naturally
      with weight_by_freq=True.
    - BackgroundMode.SUBTRACTIVE: legacy two-sample test, "this topic vs.
      all OTHER topics" (Monroe et al. style). Self-referential when the
      curriculum lives in one domain.
    """

    weight_by_freq: bool = True
    """For log-odds: post-multiply each word's z-score by its topic-internal
    relevance-weighted count. This ranks words by their per-word contribution
    to topic-keyword mutual information (≈ P(w,t) · PMI(w,t)). Crushes
    once-in-corpus tokens that pass the variance filter on small (singleton)
    topics. Set False to recover pure standardised z-scores.
    """

    bigrams_enabled: bool = True
    """Mine NPMI-thresholded bigrams from the sampled corpus and merge them
    in-place before keyword extraction. Surfaces phrases like ``bell_theorem``,
    ``single_photon``, ``photon_counting`` that are far more informative than
    their individual components.
    """

    bigrams_npmi_threshold: float = 0.4
    """Minimum NPMI for a candidate bigram. Higher = stricter (fewer, more
    confident phrases); lower = more candidates including some noise.
    """

    bigrams_max: int = 5000
    """Maximum number of bigrams kept after NPMI ranking."""

    bigrams_weight_by_freq: bool = True
    """Rank bigrams by NPMI · count (default; per-bigram mutual-information
    contribution; obviates an explicit count floor) vs. raw NPMI alone.
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
    
    method: DimensionalityReductionMethod = DimensionalityReductionMethod.LDA
    """Dimensionality reduction method for 2D projection.
    
    Options: DimensionalityReductionMethod.PCA, LDA, or UMAP
    
    - PCA: Fast, unsupervised, linear
    - LDA: Supervised, maximizes topic separation
    - LPP: Supervised, preserves local structure (generalized eigenproblem)
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


class RankMethod(str, Enum):
    PAGE_RANK = "page_rank"
    BIDIRECTIONAL_PAGE_RANK = "bidirectional_page_rank"
    CIRCUIT_RANK = "circuit_rank"


class CircuitRankSolveMode(str, Enum):
    NODE_UNSMOOTHED = "node_unsmoothed"
    NODE_SMOOTHED = "node_smoothed"
    GALERKIN = "galerkin"


@dataclass
class SimilarityGraphConfig:
    """Configuration for embedding-similarity graph smoothing."""

    enabled: bool = False
    """Whether to apply similarity graph smoothing."""

    knn_k: int = 5
    """Number of neighbors per node in the directed kNN similarity graph."""

    sharpness: float = 2.0
    """Sharpness for similarity weights: w = exp(sharpness * (cos_sim - 1))."""

    row_normalize: bool = True
    """If True, row-normalize similarity weights to sum to 1 per node."""

    density_correction_alpha: float = 5.0
    """Density correction exponent for similarity weights.

    Applies diagonal reweighting: W' = D_rho^{-alpha} W D_rho^{-alpha}, where
    rho_i = sum_j W_{i,j} (local density). Set to 0 to disable.
    """

    mix_weight: float = 0.1
    """Mixing weight for similarity edges relative to citation edges."""

    mix_mode: str = "convex"
    """Mixing strategy for citation + similarity graphs.

    - "convex": A = (1-mix_weight)*A_cite + mix_weight*A_sim
    - "additive_global_scale": A = A_cite + mix_weight * scale * A_sim,
        where scale matches total edge mass of A_cite.
    """

    include_self: bool = False
    """If True, keep self-neighbors in the kNN similarity graph."""

@dataclass
class PageRankConfig:
    """Configuration for PageRank."""
    
    alpha: float = 0.85
    """Damping factor for PageRank (probability of following a citation vs teleporting)."""
    
    tol: float = 1e-10
    """Convergence tolerance for PageRank iteration."""
    
    max_iter: int = 500
    """Maximum number of PageRank iterations."""

@dataclass
class CircuitRankConfig:
    """Configuration for CircuitRank (Currenticulum).
    
    Uses current injection at target works and optional suction at known works.
    All nodes are connected to ground via a conductance leakage, providing
    natural exponential decay without the Dirac-delta concentration issues
    of voltage clamping.
    
    Circuit equation: (D_out + G_leak * I - A) @ φ = I_sources
    Where:
    - I_sources[target] = +1/n_target (injection)
    - I_sources[known] = -1/n_known (suction, if source_work_ids provided)
    - All other nodes have I_sources = 0
    """
    
    leakage: float = 0.176
    """Global ground conductance (G_leak).
    
    Controls how quickly potential decays with graph distance.
    Lower values → more global reach (superconductor-like).
    Higher values → more local concentration (resistor network).
    """
    
    beta: Optional[float] = None
    """Alternative parameterization: leakage = exp(-beta).
    
    beta=6 → leakage≈0.002 (long-range)
    beta=10 → leakage≈0.00005 (very long-range)
    """
    
    gmres_tol: float = 1e-6
    """Convergence tolerance for GMRES solver."""
    
    gmres_maxiter: Optional[int] = None
    """Maximum GMRES iterations (None = auto)."""

    gmres_preconditioner: Optional[str] = "jacobi"
    """Optional preconditioner for GMRES. Supported: """ "'jacobi'" """."""

    gmres_log_every: Optional[int] = 50
    """If set, log GMRES progress every N inner iterations."""
    
    normalize: bool = True
    """If True, use column-normalized adjacency (PageRank mode).
    
    This makes CircuitRank equivalent to PageRank:
    - Set leakage = (1 - alpha) / alpha for damping factor alpha
    - For alpha=0.85: leakage ≈ 0.176
    - current_sources becomes the teleport/seed vector
    """
    
    use_voltage_clamping: bool = False
    """If True, use voltage clamping (Dirichlet BC) instead of current injection.
    
    Voltage clamping: Fix targets at φ=0 (ground) and (optionally) known/source works at φ=1.
    If no known/source works are provided, targets are clamped at φ=1 and the resulting
    potentials are inverted (1 - φ) for consistent ordering semantics.
    Current injection: Inject current at targets, all nodes free.
    
    With normalized adjacency, both approaches give similar distributions,
    but current injection includes the target nodes in the distribution.
    """

    edge_weighting: str = "pagerank"
    """How to weight citation edges before solving and current-flow computation.

    Modes:
    - "pagerank": existing behavior. When normalize=True, uses column-normalized
        transposed adjacency (each citer distributes unit weight across its references).
        When normalize=False, uses unweighted transposed adjacency.
    - "symmetric": symmetric degree normalization with no renormalization:
            w(v→u) = 1/sqrt(outdeg(v)*indeg(u)).
        This directly penalizes highly-cited hubs (large indeg(u)).
    - "symmetric_renorm": same as "symmetric", then renormalize per citer v so
        Σ_u w(v→u) = 1.
    - "indegree_discount_renorm": indegree discount then renormalize per citer v:
            w̃(v→u) = 1/(indeg(u)+eps)^gamma; w(v→u) = w̃ / Σ_u w̃.
        With gamma=0.5 this is a sqrt-style hub discount.

    Note: for modes other than "pagerank", the solver uses the general circuit
    Laplacian (D_out + leakage*I - A_eff) rather than the PageRank-form
    ((1+leakage)I - P).
    """

    indegree_discount_gamma: float = 0.5
    """Exponent gamma for indegree discount in edge_weighting="indegree_discount_renorm".

    Typical values:
    - 0.5: sqrt discount (variance-stabilizing)
    - 1.0: stronger discount
    """

    degree_epsilon: float = 0
    """Small epsilon added to degrees to avoid divide-by-zero in edge weighting."""

    edge_series_resistance: float = 0.0
    """Optional per-edge series resistance r (>= 0) applied to conductances.

    For each edge conductance G, the adjusted conductance is:
        G' = 1 / (1/G + r) = G / (1 + r*G)

    This penalizes longer paths by adding a fixed resistance per hop.
    Set to 0.0 to disable.
    """

    mass_transform: str = "none"
    """Optional post-solver hubness mitigation applied to node masses.

    This transform is applied after computing raw through-current masses and
    before *all* downstream steps (top-k selection, clustering, topic masses,
    keyword weighting, etc.).

    Supported:
    - "none": no transform
    - "sqrt": m' = sqrt(m)
    - "log1p": m' = log1p(m)
    - "power": m' = m**mass_power
    """

    mass_power: float = 0.5
    """Power exponent used when mass_transform="power" (e.g., 0.5 for sqrt)."""

    mass_transform_renormalize: bool = True
    """If True, rescale transformed masses to preserve the original total mass."""

    solve_mode: CircuitRankSolveMode = CircuitRankSolveMode.GALERKIN
    """How to solve CircuitRank potentials.

    - NODE_UNSMOOTHED: current behavior (citation graph only)
    - NODE_SMOOTHED: use smoothed citation+similarity graph in node space
    - GALERKIN: solve in embedding basis (current injection only)
    """

    ordering_voltage_clamp: bool = True
    """If True, compute ordering potentials in node-space when solve_mode=GALERKIN.

    This uses voltage clamping on the (possibly smoothed) graph for ordering only.
    """

    def __post_init__(self) -> None:
        if self.solve_mode == CircuitRankSolveMode.GALERKIN and self.use_voltage_clamping:
            raise ValueError(
                "CircuitRank solve_mode='galerkin' does not support voltage clamping. "
                "Use current injection (use_voltage_clamping=False) or a node-space solve mode."
            )

        if self.edge_series_resistance < 0:
            raise ValueError("edge_series_resistance must be >= 0")

@dataclass
class CurriculumConfig:
    """Configuration for curriculum building.
    
    Builds a curriculum for a target topic by mining prerequisite relationships
    using personalized PageRank or CircuitRank on the citation graph.
    """
    
    target_work_ids: List[str] = field(default_factory=list)
    """List of OpenAlex work IDs defining the target topic (seed set for PageRank).
    
    For CircuitRank: These nodes receive current injection (+I_target).
    """
    
    source_work_ids: List[str] = field(default_factory=list)
    """List of OpenAlex work IDs that the user already knows (optional).
    
    For CircuitRank: These nodes receive current suction (-I_source).
    The curriculum will prefer paths that connect targets to these known works,
    building on what the user already understands.
    
    If empty, only target injection is used with leakage to ground.
    """
    
    desired_n_topics: int = 10
    """Desired number of frontier topics in the curriculum."""
    
    # method: RankMethod = RankMethod.PAGE_RANK
    method: RankMethod = RankMethod.BIDIRECTIONAL_PAGE_RANK
    # method: RankMethod = RankMethod.CIRCUIT_RANK
    """Method to use for ranking works (PageRank or CircuitRank)."""

    similarity: SimilarityGraphConfig = field(default_factory=SimilarityGraphConfig)
    """Configuration for embedding similarity graph smoothing (available to all methods)."""
    
    pagerank: PageRankConfig = field(default_factory=PageRankConfig)
    """Configuration for PageRank."""
    
    circuitrank: CircuitRankConfig = field(default_factory=CircuitRankConfig)
    """Configuration for CircuitRank."""
    
    linkage_method: str = "bisecting_kmeans"
    """Linkage method for hierarchical clustering.

    Options:
    - 'ward', 'average', 'complete', 'single': scipy agglomerative clustering
      (requires O(n²) memory, suitable for n < 100k)
    - 'bisecting_kmeans': recursive bisecting K-Means
      (O(n) memory, suitable for n > 100k)
    - 'weighted_bisecting_kmeans': bisecting_kmeans with sample_weight from
      the post-PageRank relevance vector. High-relevance works anchor
      cluster centers; the bisection criterion respects relevance mass.
      Requires ``rank_before_clustering=True`` so a per-work weight exists.
    """
    
    min_cluster_size: int = 1
    """For bisecting_kmeans: stop splitting clusters smaller than this.

    Larger values make clustering faster but coarser.
    - 1: Split to individual points (slowest, finest)
    - 100: Stop at ~100-point clusters (faster)
    - 1000: Stop at ~1000-point clusters (much faster for large datasets)
    """

    min_cluster_weight: float = 0.0
    """For weighted_bisecting_kmeans: stop splitting clusters whose total
    sample_weight sum is at or below this floor.

    Acts in addition to min_cluster_size — whichever stop condition fires
    first wins. Lets a single high-relevance work earn singleton status by
    exceeding the weight threshold while a glob of low-relevance neighbours
    stays merged. Default 0.0 disables the weight threshold (count-only
    behaviour).
    """

    max_cluster_depth: Optional[int] = None
    """For bisecting_kmeans: maximum tree depth. None means no limit.
    
    Useful for very large datasets to limit recursion depth.
    - None: No limit (default)
    - 20: At most 2^20 = 1M leaf clusters
    - 15: At most 2^15 = 32k leaf clusters
    """
    
    rank_before_clustering: bool = True
    """If True, run ranking (PageRank/CircuitRank) first and cluster only the top-k relevant works.
    
    This is more efficient for large datasets and produces more coherent topics
    since only prerequisite-relevant works are clustered.
    """
    
    ranking_top_k: int = 5000
    """Number of top works by relevance mass to include in clustering when rank_before_clustering=True.
    
    With 100k works, Ward's linkage is tractable (~8GB memory).
    """
    
    topic_work_ids_mass_threshold: float = 0.99
    """Fraction of topic mass to retain when saving topic_work_ids.json."""
    
    topic_work_ids_max_count: Optional[int] = 50
    """Maximum number of works to retain per topic in topic_work_ids.json.
    
    If set, selection stops when EITHER the mass threshold is reached OR this count is reached.
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
    
    output_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.similarity.enabled:
            sim_cfg = self.similarity
            if sim_cfg.knn_k <= 0:
                raise ValueError("similarity.knn_k must be > 0")
            if sim_cfg.sharpness < 0:
                raise ValueError("similarity.sharpness must be >= 0")
            if sim_cfg.mix_weight < 0:
                raise ValueError("similarity.mix_weight must be >= 0")
            if sim_cfg.mix_mode not in {"convex", "additive_global_scale"}:
                raise ValueError(
                    f"similarity.mix_mode must be 'convex' or 'additive_global_scale', got {sim_cfg.mix_mode!r}"
                )


def get_curriculum_output_dir(output_dir: str, *, auto_derived: bool = False) -> Path:
    """Get the output directory for curriculum artifacts.
    
    Args:
        output_dir: Subdirectory name under artifacts/curriculum.
        auto_derived: If True, avoid collisions by appending an integer suffix
            (starting at 2) until a free directory name is found.
    
    Returns:
        Path to the output directory (created if it doesn't exist).
    """
    base = Path(ARTIFACTS_DIR) / "curriculum"
    target = base / output_dir
    if auto_derived:
        i = 2
        while target.exists():
            target = base / f"{output_dir}__{i}"
            i += 1
    target.mkdir(parents=True, exist_ok=True)
    return target


def _is_mapping(obj: Any) -> bool:
    if isinstance(obj, DictConfig):
        return True
    return isinstance(obj, Mapping)


def _to_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj, resolve=True)  # type: ignore
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Mapping):
        return obj
    # Fallback: empty mapping
    return {}


def _norm_scalar(val: Any) -> Optional[str]:
    # Convert simple values to a string form; return None to skip
    if val is None:
        return None
    # Unwrap Enums
    if isinstance(val, Enum):
        # Prefer .value if it's scalar, else use name
        sval = getattr(val, "value", val)
        return str(sval)
    if isinstance(val, (bool, int, float, str)):
        return str(val)
    return None


def _collect_overrides(curr: Any, base: Any, prefix: str = "") -> dict[str, str]:
    """Recursively collect overrides vs defaults across dataclasses or DictConfig.

    - Only include scalar fields (bool/int/float/str or Enum) to keep names short.
    - Skip lists and large nested structures.
    - Skip the 'experiment' subtree and the 'output_dir' field entirely.
    """
    diffs: dict[str, str] = {}

    if _is_mapping(curr) and _is_mapping(base):
        curr_map = _to_mapping(curr)
        base_map = _to_mapping(base)
        keys = set(base_map.keys()) | set(curr_map.keys())
        for name in keys:
            if name in {"experiment", "output_dir"}:
                continue
            sub_curr = curr_map.get(name)
            sub_base = base_map.get(name)
            key = f"{prefix}{name}" if prefix else name
            if _is_mapping(sub_curr) and _is_mapping(sub_base):
                diffs.update(_collect_overrides(sub_curr, sub_base, prefix=f"{key}."))
            else:
                sval = _norm_scalar(sub_curr)
                bval = _norm_scalar(sub_base)
                if sval is not None and bval is not None and sval != bval:
                    diffs[key] = sval
        return diffs

    # Dataclass path based on default structure (base acts as schema)
    if is_dataclass(base):
        for f in fields(base):
            name = f.name
            if name in {"experiment", "output_dir"}:
                continue
            key = f"{prefix}{name}" if prefix else name
            curr_val = getattr(curr, name, None) if not _is_mapping(curr) else _to_mapping(curr).get(name)
            base_val = getattr(base, name, None)
            if is_dataclass(base_val):
                diffs.update(_collect_overrides(curr_val, base_val, prefix=f"{key}."))
            else:
                sval = _norm_scalar(curr_val)
                bval = _norm_scalar(base_val)
                if sval is not None and bval is not None and sval != bval:
                    diffs[key] = sval
        return diffs

    return diffs


def derive_output_dir_name(cfg: CurriculumConfig) -> str:
    """Derive a default output directory name based on config overrides.

    Compares the provided config to its dataclass defaults and summarizes the
    differing scalar fields as comma-separated key=value pairs, using dotted
    paths for nested fields (e.g., "keywords.top_k=200").

    Falls back to "default" if no differences are detected.
    """
    base = CurriculumConfig()  # default instance for comparison
    diffs = _collect_overrides(cfg, base)
    if not diffs:
        return "default"
    # Prefer concise last-segment keys when unambiguous
    keys = list(diffs.keys())
    last_segments = [k.split(".")[-1] for k in keys]
    counts: dict[str, int] = {}
    for seg in last_segments:
        counts[seg] = counts.get(seg, 0) + 1
    display_keys: list[str] = []
    for k in keys:
        seg = k.split(".")[-1]
        display_keys.append(seg if counts.get(seg, 0) == 1 else k)
    parts = [f"{create_unicode_safe_filename(k)}={create_unicode_safe_filename(diffs[orig_k])}" for k, orig_k in sorted(zip(display_keys, keys), key=lambda kv: kv[0])]
    name = ",".join(parts)
    return name or "default"