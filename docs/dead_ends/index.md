# Dead ends

Ideas that were tried and refuted, with their *diagnosis*. The point is to keep failed-but-instructive directions discoverable so we don't re-run the same experiment, AND so the *reason it failed* is captured (often more useful than the original idea).

Don't dump every refuted variant here — only the *core ideas that sounded promising*. Saturation tweaks of a winning variant get a paragraph in the parent dead_ends entry, not their own files.

## Cohesion metric design

Approaches to per-gap quality measurement that were tried and refuted.

- [centroid-bayesian-likelihood-high-dim-degenerate](cohesion-metric-design/centroid-bayesian-likelihood-high-dim-degenerate.md) — fitting a Gaussian to gap members and using Bayes-mixture posterior fails in 384-dim because of the `−(d/2) log σ²` normalizing-constant term
- [global-corpus-background-degenerate](cohesion-metric-design/global-corpus-background-degenerate.md) — same root cause as above; resolved by using local CSLS-balanced background instead
- [monotone-scores-as-boundary-criteria](cohesion-metric-design/monotone-scores-as-boundary-criteria.md) — `effective_size` and `logit_sum` are monotone in size; can't be used to define where a gap stops growing
- [inverse-square-kernel-dimension-inappropriate](cohesion-metric-design/inverse-square-kernel-dimension-inappropriate.md) — `1/d²` is the 3D-specific Newtonian Green's function; in d-dim it'd be `1/r^(d-2)` ≈ delta function for d=384. Use Gaussian heat kernel instead
- [top-hat-knn-purity-loses-density](cohesion-metric-design/top-hat-knn-purity-loses-density.md) — unweighted K-NN purity loses local-density structure; weighted (Gaussian or 1/d²) preserves it
- [full-data-soft-mixture-degenerate-bootstrap](cohesion-metric-design/full-data-soft-mixture-degenerate-bootstrap.md) — `K_G/K_total` is mathematically the proper mixture posterior but degenerate from single-seed init: weight diffuses to ~|G|/N uniform, no positive feedback to grow the cluster. Size-matched local CSLS isn't just a tractability dodge; it's the calibration that makes EM converge.
- [deflation-demand-engineering-metric-artifact](cohesion-metric-design/deflation-demand-engineering-metric-artifact.md) — A-keyed/constant-s/union/gen-rank/ranking-stage demand engineering chased a phantom "book-vs-manifold tension" that was a metric misnomer (`heldout_book_recall_at_20` was full-set, not top-20, recall). Under correct recall@20 there is no tension; intrinsic `fl_bookdemand_resid` wins. Validated result: `docs/gap_deflation_results.md`

## Gap detection methods

- [exact-match-dedup-misses-near-duplicates](gap-detection-methods/exact-match-dedup-misses-near-duplicates.md) — confluence in path-dependent expansion is real but imperfect; basin-edge seeds produce ~95% identical but bit-different sets that exact-match misses but Jaccard catches

## Curriculum

- [curriculum/keyword-coclustering](curriculum/keyword-coclustering.md) — making keywords first-class in cluster *definition* (bipartite co-clustering, co-regularized 3-view, seeded bipartite-PPR) to fix vague chapters. All four strategies pin `kw_distinct`→1.0 but tank `silhouette` to ≤0.078 (mostly negative). A control showed it isn't keyword-specific: *any* multi-view blend (even the validated `[emb,cite]`) goes negative on the broad pool — embedding-direct clustering simply wins chapter cohesion. Keep clusters embedding-defined; use keywords to *label*, not define. Salvage: the nonlinear kw↔work iteration (`gap_keyword_iteration.py`) is a validated primitive for single-seed expansion; PPR's restart-anchor works (no mode-collapse).

## Shared-space alignment (wiki↔OpenAlex)

- [coocmap-anchor-space-wiki-oa](shared-space-alignment/coocmap-anchor-space-wiki-oa.md) — mycooc's coocmap (nonlinear **intra-cloud relational** re-representation, *not* VecMap) doesn't recover wiki↔OA correspondences and is beaten badly by the raw shared-encoder cosine NN (top-20 precision ~0.90 vs ~0.15–0.40; MNN 0.373→0.13, *collapses* instead of climbing). Why: the shared encoder already solves the matching (no cold-start to fix), and wiki vs OA intra-cloud relational structure isn't analogous (register residual). Positive finding: the raw cosine NN *is* the cross-corpus dictionary (~0.90); wiki-native LPP is the pedagogical direction. (Supersedes an earlier wrong-metric "mixing" refutation.)
