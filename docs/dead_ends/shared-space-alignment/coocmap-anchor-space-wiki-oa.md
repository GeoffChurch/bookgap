# coocmap anchor-space re-representation for wiki↔OA alignment — refuted

> **⚠ PROVISIONAL — RE-TESTING (2026-05-30).** This refutation is in doubt and is being
> re-run. The prototype it rests on measured **mixing** (kNN same-corpus purity → 0.5), which
> is **not coocmap's objective**: coocmap is a cross-corpus *correspondence/dictionary* method
> (unsupervised translation) — two clouds can stay locally disjoint while the matches are
> correct, exactly as English/French embedding clouds never merge yet yield a translation
> dictionary. The "Why" diagnosis below also describes an **inter-cloud** affinity, which
> contradicts the **intra-cloud** relational column-extraction coocmap actually performs
> (`Xf = M_src[:, inds1]` — each item profiles only against its *own* cloud's anchor columns;
> the cross-corpus comparison is between corresponding-anchor profile *shapes*). A corrected
> re-test (verified intra-cloud relational loop, dictionary priming, and a **correspondence**
> metric instead of mixing) is running before this is treated as settled. The "bounds CORAL/CCA
> too" implication is likewise on hold pending that re-test.

**Idea.** Borrow `~/src/mycooc`'s **coocmap** (the hard variant of softcoocmap): use
cross-domain nearest-neighbors to define **anchors**, re-represent every point in every
corpus by its similarity profile to those anchors (an "anchor space" of dim = #anchors,
summable across ≥2 corpora), recompute NNs in that space, and **iterate**. This is a
**nonlinear, non-parametric** re-representation — categorically *not* Procrustes/VecMap (an
earlier assessment mischaracterized it as an orthogonal-map loop and dismissed it for the
wrong reason; that verdict was retracted). The hope: anchor-relative coordinates could
sidestep the **covariance/nonlinear residual** that bounds the *linear* alignment methods
(CORAL/CCA/LPP-transfer/Procrustes), because "closeness" becomes "similar similarity-profile
to shared anchors" rather than "rotatable onto each other."

**Feeding path (this part works).** mycooc consumes a V×V association matrix (row i = item
i's associations to all items). The dense-embedding analog is the **gram matrix (pairwise
cosine), sparsified to kNN** — sound *in form*. One real caveat: coocmap's count pipeline
applies a variance-stabilizing transform (sqrt/PPMI/Pearson) before `unit→center→unit`; on
cosine values those transforms are **meaningless** (cosine isn't a count). The correct feed
is **raw cosine in the top-k slots → coocmap's own `unit→center→unit`**. `build_similarity_adjacency_from_*`
(`artifacts_scripts/curriculum/similarity_graph.py`) is ~90% of the builder but emits
`exp(sharpness·(cos−1))` weights, not raw cosine — bypass the sharpening if reused. With a
shared encoder the sorted-signature init is correctly skipped (identity mutual-NN ≈ 0.16, a
usable start vs ~0 for a true cross-lingual cold start).

**What was tested.** NLP-topic-matched slices (NLP-wiki top-3000 by cosine to the OA-NLP
centroid vs OA-NLP 3000), per-corpus mean-centered, k=20 kNN-gram, csls_k=10, hard coocmap
loop from the identity mutual-NN, 6 iterations, seed 0. Success metric = local kNN
same-corpus **purity** (k=10; 0.5 = fully mixed) and MMD².

**What happened — refuted.**

| stage | kNN same-corpus purity (0.5 = mixed) | MMD² |
| --- | --- | --- |
| raw centered (baseline) | 0.938 | 0.00356 |
| coocmap iter 1 → 6 | 0.976 → 0.910 | 0.00053 → 0.00031 |
| **shared-anchor upper bound** (anchors literally shared, no discovery) | **0.964** | 0.0312 |

- The iteration **does not converge to mixing**: mutual-NN rate stays stuck at ~0.15 and
  drifts *down* (0.164 → 0.149), never climbing the way coocmap does on a genuinely alignable
  pair; purity plateaus ~0.91, barely below the 0.94 raw baseline.
- The MMD² "improvement" in the matched-anchor rows is a **column-extraction artifact** —
  indexing both corpora by the same anchor pairs makes their feature rows collinear by
  construction, collapsing MMD without mixing. kNN purity (the honest metric) stays high.
- **The decisive control — the shared-anchor upper bound.** Give the method anchors that are
  *literally shared* between corpora (the best case it could ever see, anchor discovery
  removed): purity is still **0.964 — worse than the 0.938 raw baseline.** The ceiling is
  below the floor. So anchor *discovery* is not the bottleneck; the representation itself
  cannot mix these clouds.
- Induced correspondences are mostly topical noise (~2/12 sensible), the brittle-argmax
  failure on near-disjoint support.

**Why (the diagnosis — this is the artifact).** The validated wiki↔OA residual is
**near-disjoint local support** (topic-matched: kNN purity 0.92, cross-MMD 37σ, PC-subspace
angle 41° — see `docs/wikipedia-openalex-shared-space.md`). On disjoint clouds, *any* anchor
set induces **corpus-correlated profiles**: a point scores high to the anchors sitting in its
own cloud and low to the other's, so the anchor-relative coordinate **re-encodes corpus
identity instead of removing it** (and can sharpen the clumping). The covariance mismatch is
not sidestepped — it is re-expressed in anchor coordinates. The shared encoder makes this
*easier* than mycooc's hard cross-lingual case (nonzero starting MNN) but the difficulty here
is different in kind — not "no shared starting point" but "the two clouds genuinely don't
overlap locally," which no re-representation of these points repairs.

**Broader implication (bounds the rest of the mixing program).** The shared-anchor
upper-bound — *even perfect anchors can't mix them* — is evidence against the whole
**cross-corpus mixing/alignment** branch, **CORAL and CCA included**: a global second-moment
match cannot mix locally-disjoint clouds any more than perfect anchors could. Cross-corpus
*transfer* (learn on wiki, apply to OA) is bounded for the same reason — OA points sit off the
wiki manifold. The robust, encoder-level fact is that GIST encodes *register* (encyclopedic vs
academic-abstract prose) strongly enough that same-topic wiki and OA articles land in separate
local neighborhoods even after the global offset is removed.

**Salvage.**
- **None for alignment.** The cheap **per-corpus mean-centering is sufficient** for the
  use it actually serves — cross-corpus *NN retrieval* — and the topical nearest-cross-corpus
  geometry survives it (KL-ONE → its paper). No learned alignment (linear *or* this nonlinear
  one) earns its keep on the *mixing* objective.
- **The pedagogical win is wiki-native**, not cross-corpus: LPP on Wikipedia's *own* link
  graph (`docs/wikipedia-pedagogical-lpp.md`) needs no OA mixing and is unaffected by this.
- **CSLS** (the one reusable mycooc primitive) stays useful as an opt-in **NN re-ranking**
  hubness knob — already ported (`_gep_csls`) and backlogged
  (`docs/backlog/keyword-extraction/soft-hubness-correction-csls.md`); a re-ranking aid, not
  an alignment method.

**Pointers.**
- Residual characterization (the evidence): `docs/wikipedia-openalex-shared-space.md`
  (§"Residual-structure battery" + "Topic-matched re-test").
- coocmap algorithm: `~/src/mycooc/coocmap_baseline.py`, `~/src/mycooc/soft_coocmap.py`
  (`SoftCoocmapAligner`), `~/src/mycooc/signatures.py` (`get_association_matrix`). The
  count-only data model (`~/src/mycooc/data.py`) blocks direct adaptation — the only reuse is
  re-implementing the ~40-line loop, which this experiment did and discarded.
- Design context: `docs/backlog/curriculum-generalization-pedagogical-subspace.md`.
