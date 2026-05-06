# Exact-match dedup is empirically too strict for confluence

**Idea**: in `gap_article_as_seed.py`, replace Jaccard ≥ 0.5 dedup with exact-set match (`frozenset(members)` hash lookup). Theoretically motivated by confluence: if two seeds in the same basin take expansion paths that ever reach the same intermediate set, they lock and produce identical final sets. So exact-match should suffice.

**What happened**: empirically, **exact-match catches very few duplicates in practice** because path-dependence is real. Two seeds in the same basin can take different addition orderings that never produce identical intermediate sets, so they never lock — they end up with ~95% overlapping but bit-different final sets.

On T11303:
- Filtered (seed-skip + early-abort + Jaccard ≥ 0.5): **254 unique gaps**, 4 dups caught at Jaccard.
- Filtered with exact-match dedup instead of Jaccard: **258 unique gaps**, **0 dups** caught. The 4 Jaccard-caught dups are ~95% overlap but not bit-identical, so exact-match misses them.
- *Unfiltered* (no seed-skip, no early-abort, only exact-match): **789 unique gaps**, 1041 dups caught. Confirms confluence is real but imperfect.

**Theoretical analysis**:
- Seed-skip catches every case exact-match could catch (since `seed ∈ G` always; if `G == E` exactly then `seed ∈ E ⊆ claimed`). So with seed-skip enabled, exact-match dedup is *redundant* and catches 0.
- The 4 Jaccard-caught cases are basin-edge seeds: not in any prior gap (so seed-skip didn't fire), expansion drifts into existing basin, final gap is similar-but-not-identical to existing.

**Resolution**: keep Jaccard dedup (≥ 0.5) — catches the basin-edge near-duplicates exact-match misses. Or add [post-process canonicalization](../../backlog/gap-detection-methods/article-as-seed-canonicalization.md) to make path-divergent same-basin gaps converge to the same fixed point, then exact-match suffices.

**Lesson (broader)**: confluence in path-dependent dynamical systems requires that paths share an exact intermediate state, not just "be in the same basin." For confluence-based dedup to work cleanly, expansion needs a canonicalization step that's invariant to starting seed.
