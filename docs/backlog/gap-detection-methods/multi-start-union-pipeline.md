# Multi-start union pipeline for gap candidates

Combine candidate gaps from multiple methods into a single richer pool, with Jaccard-based dedup at the end. Each method explores the cohesion landscape differently:

- **Fixed-centroid radial** (`gap_kball_cohesion.py`): fastest, most conservative; gives tight cores when seed is well-centered.
- **Moving-centroid radial** (`gap_kball_moving_centroid.py`): drift handles offset K-ball seeds; finds tighter interior cores.
- **Aspherical hill-climb** (`gap_kball_hillclimb.py`): allows non-spherical extensions; catches elongated subspecialties.
- **Article-as-seed** (`gap_article_as_seed.py`): no K-ball dependence; discovers small subspecialties below K-ball's num=200 cutoff.
- **Cohesion-cut on SLT** ([backlog](cohesion-cut-on-slt.md)): tree-respecting cuts; complementary to article-centered approaches.

Different methods find genuinely different gaps in regions where the cohesion landscape has multiple local optima. The union maximizes coverage; final dedup keeps overlap reasonable.

**Pipeline**:
1. Run each method on the same topic.
2. Pool all discovered gaps.
3. Dedup with Jaccard ≥ 0.5 (or stricter), keeping the gap with highest logit_sum on collisions.
4. Rank by logit_sum or mean_logit for display.

**Risks**:
- Different methods produce gaps with different size distributions; fair ranking across them needs care.
- The union's diversity-of-shapes may include some shape-noise that no single method would have surfaced. Could combine with a quality filter (min_post ≥ 0.7, or boundary-book-relatedness).

**Test plan**: produce union on T10181, compare top-25 by logit_sum against any single method's top-25. Are the differences interpretable / improvements?
