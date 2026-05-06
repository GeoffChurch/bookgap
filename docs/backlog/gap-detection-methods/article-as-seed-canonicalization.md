# Post-process canonicalization (fixed-point iteration)

Replaces the implicit "first-seed-wins" gap definition with a **fixed-point invariant** that's seed-independent. After greedy expansion produces gap G, iteratively refine:

1. **Add** all non-G articles with `post(x | G ∪ {x}) ≥ τ`
2. **Remove** all G members with `post(i | G) < τ` (after additions)
3. Repeat until G is stable.

The fixed-point is invariant to starting seed within a basin: two seeds A and B in the same basin produce different greedy gaps G_A ≠ G_B (path-dependent), but `canonicalize(G_A) = canonicalize(G_B) = G*`. Then exact-match dedup correctly merges them.

**Why this is appealing**:
- Cheapest of the three proposed fixes (a/b/c discussed in conversation)
- Doesn't restructure the algorithm — just a polish pass on each output
- Resolves the "789 distinct exact-match basins" issue (T11303 unfiltered) by collapsing path-divergent same-basin gaps
- Foundational: makes [book-seeded confluence filter](article-as-seed-with-book-seeds.md) work cleanly with exact match

**Pairs with**:
- [Better inclusion policy](article-as-seed-better-inclusion-policy.md) — batch additions inside the canonicalization loop would make it converge faster.
- [Book-seeds](article-as-seed-with-book-seeds.md) — canonicalization makes book-vs-article confluence detection robust.
- [Lockstep expansion](article-as-seed-lockstep-expansion.md) — canonicalization is essentially the per-gap version; lockstep is the full version where all gaps share the canonicalization step.

**Test plan**:
1. Implement `canonicalize_gap(G, scope, sigma, ...)` in a new helper.
2. Run on T11303 unfiltered output (789 gaps); count distinct fixed points after canonicalization. Target: ≪ 789, closer to 254.
3. Compare canonicalized output against filtered (Jaccard-deduped) output for agreement.
