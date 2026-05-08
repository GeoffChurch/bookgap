# Lockstep expansion with same-cardinality competitors

> **See also** [multicluster-softmax-membership](../cohesion-metric-design/multicluster-softmax-membership.md) — supersedes this entry with a softmax-over-kNN-basins formulation that handles arbitrary basin sizes (no same-cardinality constraint), incorporates book-anchored basins as fixed competitors, and obsoletes most of the post-hoc enrichment pipeline. This entry retains its value as the original framing of "what's the natural background for cohesion?"

Restructure expansion from per-seed sequential greedy to all-seeds-simultaneous parallel. At each global step, every active gap expands by one (its nearest non-G non-book article); after each step, check pairwise set equality across active gaps and merge equal sets (collapse to single gap, single trajectory thereafter).

**Mechanics**:
- N seeds → N initial singletons → at step 1, every gap is `{seed, nearest_NN(seed)}`
- MNN pairs collapse at step 1; near-MNN clusters at step 2; etc.
- Continue until all active gaps are stuck (next-add would violate min_post)
- Number of distinct surviving gaps is determined by the set-equality merging process

**Natural background = the other gaps**:
At any given step, all OTHER active gaps have the same cardinality |G|. For each candidate article x and current gap G_i, the natural posterior is:
```
P(x ∈ G_i | competitors {G_1, ..., G_K \ G_i}) = K(x | G_i) / Σ_j K(x | G_j)
```
This makes the "ambient" be exactly the set of competing claims — well-defined, fair (same cardinality), and interpretable as soft K-means assignment. Confluence becomes natural: two gaps that assign the same articles ARE the same gap by definition.

**Pros**:
- Theoretically clean; resolves the "what is the ambient set?" question
- Confluence is automatic (set equality merges)
- No need for a separate canonicalization step

**Cons**:
- More compute upfront (O(N × cost_per_step) initially)
- Bigger restructuring than [canonicalization](article-as-seed-canonicalization.md) alone
- Number of steps depends on largest basin

**Test plan**:
- Prototype on T11303 (small enough to validate); compare distinct-gap count to filtered (254) and unfiltered (789) baselines.
- If it produces a single canonical gap pool of ~250, that's a strong validation.

**Variant: lockstep with book-seeds** — see [expand-from-books](article-as-seed-with-book-seeds.md). Book-seeds participate in lockstep alongside article-seeds; same-cardinality competition naturally identifies which clusters are book-anchored vs. article-only.
