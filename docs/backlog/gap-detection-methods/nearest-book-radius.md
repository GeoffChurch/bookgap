# Nearest-book-distance as the natural gap radius

Reformulate gap detection so the "radius" around a seed article is **the distance to its nearest book**, with a rank-based safety margin (e.g., take the 3/4 closest of articles within that distance). Replaces the existing geometric `radius_fraction` hyperparameter with a data-derived per-seed scale.

## The proposal

For each seed article $a$:
1. Find the distance $d_a$ to $a$'s nearest book.
2. Take the set of articles closer than $d_a$ (book-free by construction).
3. Keep the closest $\tau$ fraction (default $\tau = 3/4$) — a rank-based safety margin away from the boundary.

That's the gap candidate. No geometric radius hyperparameter, no arbitrary $\epsilon$.

## Why it might work

- **Density-adaptive by construction.** Sparse-of-books regions get large gaps, dense regions get tight ones. Reflects what we actually want gap detection to measure.
- **No arbitrary `radius_fraction`.** The 3/4 is a rank fraction (intuitive, scale-invariant), not a geometric fraction (data-dependent, instability under embedding rescaling).
- **Hubness-respecting.** Hub books shrink their nearby articles' gaps — correctly attributing coverage to the hub.
- **Aligns with the d=0 random-shuffle null** ([random-shuffle-d0-baseline] — yet to file). Under the d=0 null where each work's neighbor list is randomly ordered by counts, the expected nearest-book rank is $\approx A/B$ and the expected gap size is $3/4 \cdot A/B$. Same formula in d=0 and d>0 (the d=0 case is just the random-rank limit). Gives a clean improvement-over-random baseline for free.
- **Spiritual cousin of adaptive-$\epsilon$ KNN, OPTICS, mutual-NN graphs, and CSLS** — all use local density rather than a global radius.

## Status / working notes

**This is probably strictly inferior to several already-explored n_eff-based approaches.** A scan of `artifacts_scripts/gap_*.py` (2026-04 to 2026-05) shows the project has been moving in the density-adaptive direction for months, with multiple sophisticated alternatives:

### Closest existing analogues (use book-aware density)

- [[gap_effective_size_local]] — local-bg posterior (CSLS-balanced $K_B$ from 1-NN books + their $f(b)$-nearest articles). Uses 1-NN book information for boundary definition without explicitly using "distance to nearest book" as a radius.
- [[gap_size_metrics:kde_ratio]] — soft KDE ratio = gap density / book mixture density (spherical Gaussian fits with $\sigma_{\text{gap}}$ from member RMS-to-centroid, $\sigma_{\text{book}}$ from pairwise medians). Production-ready.

### Other density-adaptive approaches in `artifacts_scripts/`

- [[gap_effective_size]] — self-tuning Gaussian KDE purity ($\sigma_i$ = median to $K_\sigma$-NN). Per-article soft membership; `effective_size = Σ purity`.
- [[gap_effective_size_centroid]] — parametric isotropic Gaussian per gap; distance-ratio score.
- [[gap_kball_cohesion]] — K-ball centroid + radius sweep with logit-sum cohesion, auto-stops at min_post < 0.5.
- [[gap_kball_cohesion_curve]] — diagnostic per-k curve emitter.
- [[gap_kball_hillclimb]] — aspherical hill-climb on logit_sum from K-ball init.
- [[gap_kball_moving_centroid]] — moving-centroid spherical expansion.
- [[gap_soft_em_fullmix]] — full-data 2-component mixture posterior; self-tuning Gaussian.
- [[gap_size_metrics:virtual_centroid]] — production. Members preferring gap centroid over nearest book. Derives implicit radius from gap tightness vs book distances.
- [[gap_size_metrics:mutual_nn]] — production. Partition-based discount via 1-NN book + mutual-NN correction (CSLS-inspired).

### Why the nearest-book-radius proposal might still be worth noting

1. **Simplicity.** Most existing approaches involve at least one bandwidth parameter ($K_\sigma$, $K$-NN size, Gaussian fits). The nearest-book-radius formulation has only the $\tau$ (rank-fraction safety margin) and is genuinely parameter-light.
2. **Cleanest d=0 alignment.** The other approaches don't have an obvious d→0 limit that reduces to a closed-form expectation in (A, B). This one does.
3. **As a single point in the bake-off.** Could be added as a new `gap.size_metric` option in `bookgap/gap_size_metrics.py` alongside `raw_kball`, `virtual_centroid`, `mutual_nn`, `book_knn_discount`, `kde_ratio`. Comparison against the existing five would directly test whether the simplicity-vs-sophistication trade-off favors the new formulation.

## Priority

Low-to-medium. The existing methods are probably better, but this formulation is so simple it's worth running once as a bake-off arm. Specifically: implement as a new `gap.size_metric = "nearest_book_radius"` option, run on the same topics as the existing methods, see whether it lands in the top half or bottom half of the bake-off table. If it's competitive with `virtual_centroid` or `kde_ratio` despite being parameter-light, it's a publishable simplicity win. If it's at the bottom, refute and move on.

## Cross-pollination to mycooc

Mycooc's hubness work has been moving toward density-adaptive bandwidths. The nearest-class-distance-as-radius formulation may apply directly to mycooc's clustering or alignment problems where "class A vs class B" replaces "books vs articles". File under cross-pollination tracker in `denoiser-bakeoff-expansion.md` once that idea is fleshed out.
