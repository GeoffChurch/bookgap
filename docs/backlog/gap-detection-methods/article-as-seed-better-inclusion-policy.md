# Better inclusion policy for article-as-seed expansion

The strict moving-centroid expansion (`expand_seed` in `gap_article_as_seed.py`) accepts a candidate iff `min_post(G ∪ {candidate}) ≥ τ` (default τ=0.5), and stops at the first violation. This is highly path-dependent: two seeds in the same basin can take different addition orderings that never produce identical intermediate sets, so confluence (set equality) fires only ~45% of the time on T11303.

Variants to test:

- **Lower τ to 0.3 or 0.4**: paths expand further into the basin → more chance of meeting at full saturation. Path-dependence persists in principle but might empirically produce more confluence.
- **Batch additions**: at each step, add ALL candidates with `post(G ∪ {c}) ≥ τ` rather than just the nearest. Two seeds with similar growing G's would add similar batches simultaneously, increasing the chance of matching. Also more deterministic given G.
- **Mean_post or median_post objective**: replace the strict `min_post ≥ τ` floor with maximizing mean or median posterior at each step. Less brittle to single outliers; allows more expansion past local cliffs.

Pairs naturally with [canonicalization](article-as-seed-canonicalization.md): if expansion fills basins more reliably, canonicalization becomes more useful (fixed-points more likely to coincide).

Quick test: re-run `gap_article_as_seed` on T11303 with `--min-post-threshold 0.3` and check whether the unfiltered exact-match pool of 789 collapses meaningfully (toward the filtered 254).
