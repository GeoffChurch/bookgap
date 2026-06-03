# Soft topic membership: vMF-mixture-with-null + per-work bandwidth (boundary-background for soft)

**Origin (conversation 2026-06-03).** Design review of the validated embedding-soft keyword
membership ([[curriculum-fuzzy-soft-topic-membership]]) asked whether the *boundary/local-background*
principle that won elsewhere (the gap-selection `log p/g_local`; the curriculum-marginal keyword lift
instead of the global 6.8M corpus) is applied inside `soft_responsibilities`. It is **not**, and that's
the clearest improvement.

## The gap: the softmax has no ∅/background sink

`gap_fuzzy_topics.soft_responsibilities` returns `softmax(sims/τ, axis=1)` — rows sum to 1 **over the N
chapters**, no "belongs to none" outlet. So a work far from *every* chapter (a foundation-wander
intrusion — IoT, profilometry; or Color Image's catalog-blurb) still dumps its full unit of mass into
its least-far chapters and pollutes them. `soft_floor` (0.05, applied downstream as `resp >= floor`)
does **not** substitute for a sink: a *diffuse* work whose responsibilities spread ~uniformly
(≈1/13 = 0.077 each) **exceeds the floor on many chapters at once → contributes to all of them** (max
pollution), while a peaked work contributes to one. This is precisely the increment-1 Color Image
failure (soft pulled in *more* boilerplate) and the soft analog of what gap-selection already does
right: commit only above the local ambient.

## The principled form: vMF mixture **with a uniform-background component**

`softmax(κ·cosine)` already *is* a von Mises–Fisher mixture posterior — just missing the null. Make it
the full generative model: N vMF components (the chapters) **+ one uniform/background component on the
sphere** (the ∅ sink). The responsibility is the proper mixture posterior; a work whose best
chapter-similarity doesn't beat the **ambient** similarity lands mostly in ∅ and contributes ≈nothing.

- **∅ logit = ambient/boundary similarity.** Natural references: the work's cosine to the **pool
  centroid** (the local ambient direction), or a fixed high quantile of the work's own
  sim-to-chapter distribution, or a size-matched local-background expectation (the `g_local` analog).
  This is the boundary-background transfer: a work earns chapter membership only by being *more*
  chapter-like than the local null.
- The absorbing-walk arm in [[curriculum-fuzzy-soft-topic-membership]] gets this **free** via the
  `(1−α)` teleport leak; the softmax arm (the validated one) just never had it added. Closing that is
  the single highest-leverage change — it directly fixes diffuse/boilerplate works polluting chapters.

## Same-family refinements

- **Per-work τ (t-SNE bandwidth), not one global τ.** `_calibrate_tau` finds a *single* τ for the
  *mean* perplexity ≈ 1.5; the "per-work adaptivity" the backlog credits for embedding-soft beating
  citation-soft is only approximate (heterogeneous perplexity falling out of a global τ). Per-work τ
  (each work hits ~1.5 like t-SNE's per-point β) makes dense- and sparse-region works commit by
  *relative* geometry, not absolute scale. Cheap at N=13 (bisect per row). This is the κ-per-element
  analog of the ∅ idea (per-component or per-work concentration).
- **Robust anchors for thin chapters.** Responsibility centroids are hard-cluster *means*; a 1-work
  cluster's centroid *is* that one (often boilerplate) work — so soft re-attraction chases the blurb
  (Color Image). Use a top-mass **core set / medoid** per chapter (the absorbing-walk design's `S_k`),
  or shrink thin-cluster centroids toward the pool mean.

## Where the same idea helps other aspects

- **dynamic-k floor → per-topic ESS cut.** The keep-threshold is a global relative floor (0.05).
  Replace with the topic's **effective size** `ESS = (Σ mass·r)² / Σ(mass·r)²` so a genuinely thin
  topic honestly shows few keywords and a rich one many — no global knob (the participation-ratio
  already surfaced as `summary.md`'s "Effective # of Works"). Flagged in
  [[curriculum-fuzzy-soft-topic-membership]].
- **Seeded bipartite-PPR edges are already degree-normalized** (keyword degree = document frequency),
  so generic-keyword hubness is damped to first order; **lift-weighting the edges** (`B[d,t] = log-lift`
  not binary) would bind works to their *distinctive* keywords — folding the boundary-background lift
  into the graph. Lower priority (PPR was refuted on silhouette in
  `docs/dead_ends/curriculum/keyword-coclustering.md`).

## Eval

The ∅ sink should: (a) leave peaked works' responsibilities ~unchanged (argmax preserved → generalizes
the hard pipeline still holds); (b) move diffuse/off-topic works' mass to ∅ (mean topics/work drops for
the intrusions, not the core); (c) on Color Image, dissolve the blurb chapter's boilerplate *without*
re-attracting more (the increment-1 failure inverts). Judge via the ToC-reconstruction panel +
`gap_intrinsic_metrics.py` (watch kw_gap_pmi / kw_distinct rise without silhouette dropping). Validate
`argmax(r) ≈ hard labels` first.
