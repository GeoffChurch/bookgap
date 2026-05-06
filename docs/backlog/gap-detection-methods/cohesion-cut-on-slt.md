# Cohesion-cut on the single-linkage tree

Use HDBSCAN's single-linkage tree (already computed and saved by `hdbscan_gap_probe --save-state-dir`) but replace max-bookless's "stop walk-up at first ancestor with a book" rule with a cohesion-aware cut criterion.

**Recipe**:
1. For every internal node v in the SLT, score `effective_size_local(articles_under_v)` using the local-bg formulation.
2. Greedily pick a non-overlapping set of nodes that maximizes total effective size (max-weight independent set on the tree — linear-time greedy because tree-structured constraints).
3. Each chosen cut is a candidate gap.

**Why this is interesting**:
- Reuses existing infrastructure (SLT is already computed for max-bookless)
- Generalizes max-bookless: the "any book → stop" rule is a special case where the cut criterion is binary book-presence; this generalizes to a continuous cohesion-aware criterion.
- For book-poor topics (T13734) where max-bookless produces sprawling 5,325-article gaps, cohesion-cut would naturally prefer the tighter sub-clades that have higher per-node effective_size.

**Cost**: cheap. SLT has ~N nodes; scoring each is O(K_query) on cached embeddings; max-weight independent set is linear.

**Comparison to article-as-seed**:
- Tree cuts respect the dendrogram order; article-as-seed doesn't.
- For book-rich topics with clean cluster hierarchy (T11303 BNs), tree cuts and article-as-seed should agree.
- For topics where the SLT has incidental near-duplicate-driven early-merges (T10181's templated-data leaks), tree cuts inherit the SLT's structural quirks; article-as-seed is shape-agnostic and may find cleaner groupings.

**Test plan**:
1. Score every internal SLT node on T10181 (using existing state-dir + local-bg).
2. Run greedy max-weight independent-set selection.
3. Compare the resulting gap pool to article-as-seed and max-bookless on the same topic. Identify gaps unique to each method — those are the "shape-dependent discoveries."
