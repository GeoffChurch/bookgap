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

## Gap detection methods

- [exact-match-dedup-misses-near-duplicates](gap-detection-methods/exact-match-dedup-misses-near-duplicates.md) — confluence in path-dependent expansion is real but imperfect; basin-edge seeds produce ~95% identical but bit-different sets that exact-match misses but Jaccard catches
