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
    """Number of top keywords per topic (upper cap)."""

    dynamic_topk_floor: float = 0.0
    """Dynamic top-k: keep only keywords scoring >= this fraction of the topic's
    TOP keyword score (0.0 = off, fixed top_k). Self-calibrating per topic —
    a thin/degenerate chapter whose keyword scores cliff to ~0 after a few real
    terms shows just those, instead of padding top_k with ~0-score junk. A healthy
    chapter (scores decay smoothly) keeps the full top_k. ~0.05 is a good default."""

    dynamic_topk_min: int = 3
    """Floor on the number of keywords shown per topic when dynamic_topk_floor>0."""

    soft_topics: bool = False
    """Fuzzy keyword extraction: instead of assigning each work to its single hard
    topic, give it a soft responsibility r_{i,k} (one-shot softmax over topic
    centroids) and let it contribute to every topic with r >= soft_floor, weighted
    by mass*r. Dissolves degenerate ~1-work chapters whose keywords would otherwise
    be one work's idiosyncratic tail (validated on Neuromorphic: a lone-VCSEL-paper
    chapter became a clean 30-keyword VCSEL chapter, and the judge re-read ours as
    the more focused curriculum). See docs/backlog/curriculum-fuzzy-soft-topic-membership.md.

    OPT-IN (default off) because it is **~5x slower**: soft pools are larger (each
    work enters multiple topics' phrase-mining pools), so keyword extraction went
    from ~3 min to ~15 min on a 13-topic / 5000-work run. It is *probably* a quality
    improvement (cleaner, more specific chapters), but the speed cost isn't always
    worth it — enable it deliberately, especially on debris-/degenerate-prone runs.
    `soft_floor` bounds the pool size."""

    soft_mode: str = "embedding"
    """How soft responsibilities are computed when soft_topics=True:
    - 'embedding': one-shot softmax over EMBEDDING cosine to topic centroids.
    - 'citation': citation-aware label propagation (the absorbing-walk arm) —
      diffuse the hard labels through the citation graph. More citation-coherent
      (the FL/DPP refutation showed citation-coherence drives distinctive chapters)."""

    soft_alpha: float = 0.5
    """For soft_mode='citation': citation-diffusion weight (1-alpha teleports back
    to the work's own hard label — the anti-diffusion clamp)."""

    soft_alpha_mode: str = "global"
    """For soft_mode='citation': 'global' uses soft_alpha everywhere; 'adaptive'
    sets per-node alpha = 1 - (fraction of citation neighbors sharing the node's
    hard label) so coherent nodes keep their label and boundary nodes diffuse."""

    soft_class_balance: bool = False
    """For soft_mode='citation': class-mass normalization (upweight small/degenerate
    topics so they win recruits)."""

    soft_perplexity: float = 1.5
    """Target mean topics-per-work for the softmax temperature (t-SNE-style)."""

    soft_floor: float = 0.05
    """A work contributes to a topic's keyword pool only if its responsibility to
    that topic is >= this (keeps pools focused; the soft weight is mass*r)."""

    soft_background: str = "none"
    """For soft_mode='embedding': the canonical ∅/background sink (vMF-mixture-with-null).
    Appends the pool's mean DIRECTION μ₀ as a (K+1)th vMF component at the same κ, so the
    responsibilities become a softmax over the ambient-subtracted similarities (c_ik − c_i0):
    a work whose every chapter-cosine is below its ambient-cosine leaks to ∅ and contributes
    nowhere, instead of force-distributing its mass and polluting chapters (the boundary-
    background move; the probabilistic --mean-center). Options:
    - 'none'  (default): today's pure-simplex softmax (rows sum to 1, no ∅).
    - 'mass'  : μ₀ = mass-weighted pool centroid (where the typical *important* work points;
      consistent with the mass*r weighting everywhere else).
    - 'mean'  : μ₀ = plain (unweighted) pool centroid (importance-agnostic geometric ambient).
    See docs/backlog/curriculum-soft-membership-background-sink.md."""

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
    """Mine adjacent-token phrases via the lattice + combined-score recipe
    in ``keyword_extraction.mine_phrases_lattice``: each candidate is scored
    by ``NPMI(a,b) · count(a,b) · min(left_be, right_be)`` and must clear
    all three thresholds. Surfaces phrases like ``bell_theorem``,
    ``single_photon``, ``photon_counting``, ``einstein_podolsky_rosen``
    while rejecting boilerplate aggregates (``articles_citing``,
    ``manager_easybib``) that NPMI alone admits.
    """

    bigrams_npmi_threshold: float = 0.4
    """Minimum NPMI for a candidate bigram. Higher = stricter (fewer, more
    confident phrases); lower = more candidates including some noise.
    """

    bigrams_max: int = 5000
    """Maximum number of phrases kept *per iteration* (the cumulative phrase
    vocabulary across iterations can be larger).
    """

    bigrams_min_boundary_be: float = 0.5
    """Absolute boundary-entropy floor in bits. The boundary-entropy
    filter rejects fixed-sequence boilerplate (``articles citing this
    article``: low entropy on both sides because the surrounding
    context is always the same) while keeping genuine phrases
    (``bell theorem``: high boundary entropy because lots of different
    things precede and follow it). 0.5 is empirically the floor that
    rejects ~13/14 known boilerplate items in the `__24` probe while
    retaining 7/11 known good phrases.

    Use *together with* ``bigrams_be_quantile`` (combined via ``max``).
    The absolute floor stabilises behaviour across iterations of the
    lattice loop, where the BE distribution shifts as merges consume
    candidates.
    """

    bigrams_be_quantile: float = 0.0
    """Optional data-driven BE threshold expressed as a quantile of the
    candidate BE distribution. ``0.0`` (default) disables it; any
    positive value ``q`` requires candidates to clear both the absolute
    ``bigrams_min_boundary_be`` floor *and* the ``q``-th percentile of
    candidate BEs.

    The quantile alone shifts unpredictably across iterations (median
    BE drops as merges consume varied candidates, admitting boilerplate
    aggregates in iter 2+ that the absolute floor catches). Mostly
    useful as a stricter top-up — e.g. ``q=0.75`` to keep only the
    upper quartile — when the absolute floor isn't tight enough.
    """

    bigrams_normalize_be: bool = False
    """If True, the BE factor in the phrase score becomes
    ``BE / log2(count + 1)`` (count-normalised, bounded roughly in
    ``[0, 1]``) rather than absolute BE in bits. Each candidate is
    scaled by its own count, which is intrinsically iteration-stable —
    a candidate's normalised score doesn't depend on the population
    distribution shifting between iterations. Threshold filters
    (``bigrams_min_boundary_be`` and ``bigrams_be_quantile``) still
    apply to the un-normalised BE.
    """

    bigrams_count_floor: int = 3
    """Minimum count per candidate. Small floor is unavoidable for stable
    boundary-entropy estimates (a 1-occurrence bigram has entropy 0 by
    construction).
    """

    bigram_iterations: int = 6
    """Maximum iterative phrase-mining passes. Each pass scores bigrams over
    the *current parse*, adds new ones to the cumulative phrase vocab, and
    re-parses the raw corpus with longest-match against the updated vocab.
    Self-terminates when an iteration adds no new phrases. iter 1 → up to
    bigrams, iter 2 → up to trigrams (e.g. ``einstein_podolsky_rosen``
    forms when iter 1's ``einstein_podolsky`` + ``rosen`` is mined as a
    bigram in iter 2 over the iter-1 parse), iter N → up to (N+1)-grams.
    Default 6 is a generous cap; typical corpora exit after 2–3 passes.
    Re-parsing each iteration implements the semtok-style "exclusive
    entropy" effect: counts attributable to longer phrases never appear
    in shorter-phrase context distributions, because the longest-match
    parser consumes those positions on every pass.
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
    
    desired_n_topics: int = 15
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
    
    linkage_method: str = "weighted_bisecting_kmeans"
    """Clustering method for the curriculum.

    Options:
    - 'ward', 'average', 'complete', 'single': scipy agglomerative clustering
      (requires O(n²) memory, suitable for n < 100k)
    - 'bisecting_kmeans': recursive bisecting K-Means
      (O(n) memory, suitable for n > 100k)
    - 'weighted_bisecting_kmeans': bisecting_kmeans with sample_weight from
      the post-PageRank relevance vector. High-relevance works anchor
      cluster centers; the bisection criterion respects relevance mass.
      Requires ``rank_before_clustering=True`` so a per-work weight exists.
    - 'leiden': Leiden community detection on the *citation subgraph* of
      the top-k relevance-filtered works. Citations are the semantic
      relation curricula *means* (one paper cites another because it
      builds on it), so community detection on the citation graph
      produces a topic structure based on the underlying relation
      rather than on embedding distance. Communities directly serve as
      frontier topics — no embedding-based hierarchical merging or
      mass-threshold cut. Tune ``leiden_resolution`` to control the
      community count (higher → more, smaller communities). On
      tightly-knit corpora the count is hard to tune precisely;
      consider 'spectral' if exact-N control matters. Requires
      ``rank_before_clustering=True``.
    - 'spectral': spectral clustering on the citation subgraph of the
      top-k filtered works. Symmetrises adjacency, computes the
      *K* smallest eigenvectors of the normalised Laplacian (where
      *K* = ``desired_n_topics``), and runs K-means in the embedding.
      Same citation-graph signal as 'leiden' but with *K* as a direct
      parameter (Leiden's ``resolution`` is data-dependent and often
      hard to tune precisely). Quality on tightly-knit citation
      subgraphs is generally close to Leiden's; on weakly-connected
      graphs Leiden tends to be more robust. Requires
      ``rank_before_clustering=True``.
    - 'graph_bisect': recursive Fiedler-based bisection on the citation
      subgraph. At each step picks the largest current cluster and
      splits it via the second-smallest eigenvector of its induced
      subgraph's normalised Laplacian (median split). Recurses until
      *K* clusters exist. Different bias from 'spectral': each
      bisection adapts to the local subgraph structure rather than
      relying on a single global eigenvector decomposition; produces
      a clean recursive hierarchy. Direct *K* control. Requires
      ``rank_before_clustering=True``.
    - 'citation_avg_linkage': agglomerative average-linkage on a
      Jaccard distance between citation neighbourhoods, with the
      existing mass-threshold cut producing the frontier. Citation-
      based (like 'leiden'/'spectral'/'graph_bisect') *and* allows
      earned singletons (like 'weighted_bisecting_kmeans'/'average'),
      because the mass-threshold cut promotes high-mass works whose
      neighbourhood is distinctive enough that no merge brings them
      under the threshold. The closest direct analogue of
      'weighted_bisecting_kmeans' on the citation graph instead of
      the embedding space. Requires ``rank_before_clustering=True``.
    - 'multi_view_spectral': co-regularized multi-view spectral
      clustering combining an embedding-similarity view (cosine on
      denoised embeddings) and a citation-similarity view (Ruzicka on
      WU-mode neighbourhoods). CSLS hubness correction is applied to
      each view before forming its Laplacian. Iterates eigendecomposition
      of ``L_v - λ U_w U_wᵀ`` until subspace convergence; K-means on the
      row-normalised concatenated joint embedding produces the labels.
      ``λ=0`` recovers per-view independent spectral. Reference:
      Kumar-Rai-Daumé (2011), Conneau et al. (2017). Requires
      ``rank_before_clustering=True``.
    - 'facility' / 'dpp': pick ``desired_n_topics`` DIVERSE anchor works
      (greedy facility-location max-coverage / greedy MAP-DPP over the
      top-``anchor_candidates`` works by mass), then assign each work to its
      nearest anchor. Directly de-duplicates topics — the dendrogram cut makes
      every topic pair redundant (cos>0.7) on tight corpora; FL/DPP anchors are
      mutually dissimilar. Pairs naturally with ``keywords.soft_topics=True``.
      Requires ``rank_before_clustering=True``. See
      docs/backlog/curriculum-fuzzy-soft-topic-membership.md.
    """

    anchor_candidates: int = 400
    """For linkage_method 'facility'/'dpp': number of top-mass works considered
    as candidate anchors."""

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

    leiden_resolution: float = 1.0
    """For ``linkage_method='leiden'``: resolution parameter of the
    Reichardt-Bornholdt configuration null model. Higher values produce
    more, smaller communities; lower values fewer, larger ones.

    Tune to match ``desired_n_topics``: try 1.0 first, then bump up if
    too few communities, down if too many. There's no exact mapping —
    Leiden picks the count that best optimises modularity at the given
    resolution. Reasonable range: 0.5 to 3.0.
    """

    citation_jaccard_mode: str = "binary"
    """For ``linkage_method='citation_avg_linkage'``: how to weight the
    citation neighbourhood before computing pairwise similarity.

    Options:
    - 'binary' (default, original): unweighted in+out neighbourhood,
      ordinary Jaccard.
    - 'weighted_unified': weight each neighbour by its edge weight in
      the row-normalised random-walk transition matrix
      ``P[i, j] = A[i, j] / out_deg(i)``; neighbourhood is
      ``max(P, P^T)``. Generalised Jaccard (Ruzicka). Down-weights
      promiscuous citers (literature reviews) automatically.
    - 'weighted_per_direction': separately ℓ1-normalise the
      out-distribution (row-normalised P) and the in-distribution
      (column-normalised A) per node, concatenate, then Ruzicka.
      Decouples out-degree noise from in-degree noise.
    """

    multi_view_spectral_co_reg_lambda: Optional[float] = None
    """For ``linkage_method='multi_view_spectral'``: the co-regularisation
    coupling strength λ between the two views (embedding and citation).

    If ``None`` (default), the solver auto-tunes λ from the Laplacian
    spectra via
    ``auto_lambda_scale * (op_norm(L_emb) + op_norm(L_cite)) / (2 * K)``,
    a heuristic empirically validated cross-corpus on QC quantum-crypto
    and vision→robotics curricula (both had Q-optimum at λ=0.1; auto-tune
    landed within −0.002 of optimum on robotics, −0.019 on QC). The bulk-
    spectrum op-norm proxy is robust to bottom-K eigenvalue degeneracy
    (common on near-rank-deficient Laplacians from CSLS-clipped similarity
    matrices, where K-th-eigenvalue heuristics collapse to numerical noise).

    Set to a numeric value to override. ``0.0`` ≈ per-view independent
    spectral; large values force agreement (degenerate above ~5).
    """

    multi_view_spectral_auto_lambda_scale: float = 1.5
    """For ``linkage_method='multi_view_spectral'``: multiplicative constant
    in the auto-tune formula
    ``λ = scale * (op_norm(L_emb) + op_norm(L_cite)) / (2 * K)``,
    applied only when ``multi_view_spectral_co_reg_lambda`` is ``None``.
    Default ``1.5`` was empirically validated cross-corpus (QC and
    robotics): both corpora had op_norm_per_K ≈ 0.065 and Q-optimum at
    λ=0.1, so 1.5 × 0.065 ≈ 0.098 — essentially the optimum.
    """

    multi_view_spectral_csls_k: int = 10
    """For ``linkage_method='multi_view_spectral'``: the K used in CSLS
    hubness correction's mean-of-top-K-neighbours computation, applied
    to each view before forming its Laplacian.

    Larger K averages over more neighbours and produces smoother
    correction; smaller K is more sensitive to immediate hubness.
    Reasonable range: 5 - 30 for n ~ thousand-scale.
    """

    multi_view_spectral_citation_mode: str = "weighted_unified"
    """For ``linkage_method='multi_view_spectral'``: which citation
    Jaccard mode to use for the citation view. Defaults to
    ``weighted_unified`` because its row-normalisation already de-hubs
    the out-direction; CSLS then complements with in-direction
    de-hubbing. See ``citation_jaccard_mode`` for the full set.
    """

    multi_view_spectral_max_iters: int = 30
    """For ``linkage_method='multi_view_spectral'``: maximum alternating-
    eigendecomposition iterations. Convergence is usually reached in
    fewer than 10 iterations on well-conditioned data.
    """

    multi_view_spectral_tol: float = 1e-5
    """For ``linkage_method='multi_view_spectral'``: subspace-distance
    convergence tolerance (Frobenius norm of the change in projector
    ``U_v U_vᵀ`` between iterations).
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

    background_normalize: bool = False
    """If True, rank the top-k SELECTION by a background-normalized score
    ``p^a * log((p+eps)/(g+eps))`` instead of raw PR mass, where ``p`` is the
    foreground (gap-seeded) PR and ``g`` is a global (uniform-teleport) PR. Demotes
    popularity-explained mass (off-topic global hubs) while keeping works the gap
    specifically elevates (``p/g > 1``). Cluster weights stay raw PR mass.
    See docs/backlog/curriculum-ranking-input-levers.md. PageRank methods only."""

    background_exponent: float = 0.5
    """Exponent ``a`` in ``p^a * log(p/g)``. a=0 = raw log-lift; a=0.5 =
    Fightin'-Words variance-normalized (canonical default, trust ∝ support);
    a=1 = pointwise KL / Bayesian surprise."""

    background_alpha: Optional[float] = None
    """Damping for the background global PR. ``None`` => match the foreground
    ``pagerank.alpha`` (canonical default: same kernel, teleport-only contrast).
    Higher => coarser baseline that discounts more global/universal mass."""

    background_eps: float = 1e-12
    """Laplace floor in ``log((p+eps)/(g+eps))``; guards p=0 nodes (matters for a=0)."""

    boundary_selection: bool = False
    """If True, rank the top-k SELECTION by a *boundary-background*-normalized
    score ``p^a * log((p+eps)/(g_local+eps))``, where ``p`` is the foreground
    (gap-seeded) forward PR and ``g_local`` is the *field-local* background from
    two extra PageRank passes (backward-from-p, then forward-from-that) rather
    than the global uniform-teleport baseline. Demotes popularity hubs whose mass
    is explained by their own off-topic citers, while keeping field foundations.
    Uses ``background_exponent`` (a), ``background_eps`` (eps), and
    ``pagerank.alpha``. Mutually exclusive with ``background_normalize``.
    PageRank methods only. See artifacts_scripts/curriculum/pagerank.boundary_background."""

    semantic_rerank: bool = False
    """Product-of-experts SELECTION: rank the top-k by
    ``log((p+eps)/(g_local+eps)) + log f_hat(v)`` — the field-local excess
    centrality (same ``g_local`` as boundary_selection) times a Gaussian KDE of
    the gap seeds evaluated at each work. Combines two independent relevance
    signals (citation importance and semantic proximity to the gap) as a product;
    demotes citation-central but semantically-distant foundations. Takes priority
    over boundary_selection/background_normalize. PageRank methods only."""

    semantic_bandwidth: Optional[float] = None
    """KDE bandwidth h^2 in squared-distance units (||v-s||^2 = 2(1-cos) on
    unit-normalized embeddings). None => median pairwise squared distance among
    the seeds (the median heuristic; data-driven, no tuned knob)."""

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