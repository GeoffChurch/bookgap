# Keywords for cluster DEFINITION (bipartite co-clustering) — refuted

**The idea.** The dendrogram curriculum's one soft spot is occasionally *vague/generic/overlapping*
chapters. Keywords are exactly what a chapter is "about", so make them **first-class in clustering**,
not just a post-hoc label: cluster the works so that each chapter is *keyword-coherent by
construction*. Concretely, co-cluster the work×keyword bipartite graph (LSA/NMF/spectral), or run a
seeded diffusion that defines a chapter as {works + characteristic keywords} jointly.

**Verdict: refuted.** Every strategy that lets keywords *define* clusters maximises keyword
distinctiveness (`kw_distinct` → 1.000, the target defect nominally "fixed") but **destroys embedding
cohesion** — `silhouette` collapses from the dendrogram's 0.192 to ≤0.078, mostly negative (works end
up closer to other chapters' centroids than their own). And a control showed the cause is **not even
keyword-specific**: it's that *any* multi-view blend trades embedding cohesion for cross-view
agreement on the broad pool. Embedding-*direct* clustering is simply the right tool for chapter
tightness.

## What was tested (neuromorphic field-17, 13 chapters, 5000-work pool)

| method | code | silhouette | kw_distinct | kw_gap_pmi | prereq |
| :-- | :-- | :--: | :--: | :--: | :--: |
| **dendrogram `_sem` (validated stack)** | `build_curriculum.py` | **0.192** | 0.908 | 3.635 | 0.710 |
| book ToC (reference) | `book_toc_reconstruction.py` | 0.154 | 0.952 | 4.210 | 0.534 |
| inc 1 — nonlinear kw↔work iteration | `gap_keyword_iteration.py` | *(primitive — validated, see salvage)* | | | |
| inc 2 — keyword-deflation curriculum | `gap_keyword_curriculum.py` | 0.078 | 1.000 | 2.713 | 0.678 |
| inc 3a — kernel-sum + graph-bisect | `gap_multiview_term.py --method kernelsum` | −0.024 | 1.000 | 2.836 | 0.561 |
| inc 3b — co-regularized 3-view | `gap_multiview_term.py --method coreg` | −0.038 | 1.000 | 2.419 | 0.619 |
| seeded bipartite-PPR | `gap_seeded_ppr.py` | −0.043 | 1.000 | 2.229 | 0.624 |

(kernel-sum via `SpectralClustering(precomputed)` additionally *degenerated* to one giant cluster
[4975, 3, 2…]; graph-bisect forced balance but stayed incoherent.)

**Expected:** a balanced point — keyword info would *raise* `kw_distinct`/gap-fidelity while the
embedding/citation views *held* `silhouette` near the dendrogram's 0.192. **Got:** `kw_distinct`
pinned at 1.0 every time, `silhouette` at or below zero every time. The "fix the vague chapter"
distinctiveness was achieved precisely *by* arranging clusters along keyword-space axes, which are
~orthogonal to embedding cohesion.

## The control that corrected the diagnosis

The obvious read of inc-3b was "the term view is anti-cohesive." A control over the **same 5000-work
pool** (silhouette in embedding space — emb-only KMeans vs 2-view vs 3-view co-reg, reproducible via
`gap_multiview_term.py --diagnose-cohesion`) refuted *that* too:

```
emb-only KMeans             : +0.237   (embedding-direct → cohesive)
2-view coreg [emb,cite]     : -0.052   (the VALIDATED two-view — also negative!)
3-view coreg [emb,cite,term]: -0.038   (term view marginally HELPS over 2-view)
```

So the term view is **not** the culprit — the *citation* view already drives the co-reg clustering
negative, and adding keywords slightly *improves* it. The real mechanism: co-regularized multi-view
spectral clusters in the **joint** spectral space `[U_emb | U_cite | U_term]`; any view that is
anti-aligned with embedding geometry pulls the final k-means off the embedding-cohesive partition. On
the broad pool (which carries foundation-wander + off-topic intrusions) the citation view *is* that
anti-aligned view. Embedding-direct clustering (KMeans 0.237, dendrogram bisecting-kmeans 0.192) wins
cohesion because it optimises cohesion directly.

**No contradiction with the validated 2-view co-reg in `build_curriculum`:** that path was tuned on
**modularity-Q** (a citation-structure objective), on the much tighter *selected* set (~368 works
post product-of-experts), where the emb and cite views both concentrate on the gap and agree. On the
full 5000-pool the views diverge, and the embedding-cohesion criterion (silhouette) — a *different*
objective — goes negative. Multi-view isn't "broken"; it optimises cross-view agreement, which is
simply not the chapter-tightness criterion.

## Why `kw_distinct = 1.0` isn't worth chasing

The dendrogram already scores `kw_distinct = 0.908` — **near max**. The headroom to 1.0 is tiny, and
buying it costs *all* the embedding cohesion (0.192 → negative) that makes chapters semantically
tight. The vague-chapter defect is better attacked **upstream and at labelling time**, not in the
partition:

- **Upstream:** the product-of-experts selection (`log p/g_local + log f̂`) removes the foundation-
  wander / off-topic intrusions (IoT, fringe-profilometry, policy-transfer basins kept surfacing in
  the keyword runs) that *cause* vague chapters in the first place.
- **At labelling time:** dynamic-top-k keyword extraction *describes* the embedding-defined clusters
  cleanly (and the keyword runs' debris — Spanish stopwords from a Spanish seed title, `sup_xmlns`
  XML markup from abstracts — shows keyword hygiene is its own problem the partition shouldn't depend
  on).

**Keep clusters embedding-defined; use keywords to label, not to define.**

## Salvage paths

- **`gap_keyword_iteration.py` (inc 1) is a VALIDATED primitive — keep it.** The *nonlinear*
  keyword↔work fixed-point iteration (run the real log-lift keyword extraction in the loop, not a
  linear map) converges to seed-dependent, coherent, keyword-clean basins (Jaccard 0.05 between two
  distinct seeds' fixed points — the nonlinearity beat linear mode-collapse with **no** anchor). It's
  the right tool for *interactive single-seed expansion* ("grow this seed into its keyword-coherent
  neighbourhood") — just **not** for partitioning a whole pool into a curriculum.
- **Seeded bipartite-PPR's restart works as an anchor.** `gap_seeded_ppr.py` confirmed the recurring
  anti-diffusion lesson's fix: the teleport term pins each basin to its seed (no mode-collapse), so
  the *linear* walk produced coherent per-seed basins too — they were just keyword-coherent, not
  embedding-coherent. The anchoring mechanism is sound and reusable; the basis (keyword space) is the
  problem.
- **If keyword-coherent AND embedding-coherent is ever wanted:** cluster in embedding space *first*,
  then split/merge only by keyword distinctiveness as a tiebreaker — a post-hoc refinement of an
  embedding partition, never a co-equal view. Untested; noted.

## Recurring lesson

Third confirmation of the **anti-diffusion** lesson (cf.
`cohesion-metric-design/full-data-soft-mixture-degenerate-bootstrap.md` and the citation-soft/
absorbing-walk results in the fuzzy-soft-topic-membership backlog): plain diffusion forgets the seed;
a restart/teleport anchor restores per-seed basins. New lesson here: **co-clustering criteria are not
interchangeable.** Embedding cohesion, citation modularity, and keyword distinctiveness are mutually
~orthogonal objectives on a broad corpus; a blend optimises none of them, and for the
curriculum-chapter-tightness goal the embedding-direct partition dominates.

---
*Tested 2026-06-02. Producers: `artifacts_scripts/curriculum/gap_keyword_iteration.py`,
`gap_keyword_curriculum.py`, `gap_multiview_term.py` (`--method {coreg,kernelsum}`),
`gap_seeded_ppr.py`; eval `gap_intrinsic_metrics.py --run-a <run> --run-b
neuromorphic_trackA_sem --gap-json artifacts/book_gaps/neuromorphic_gap.json --field 17`;
isolation control `gap_multiview_term.py --pool-run-dir
artifacts/curriculum/neuromorphic_trackA_sem --field 17 --diagnose-cohesion`. Regenerate
rather than edit by hand.*
