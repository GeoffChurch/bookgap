# coocmap anchor-space re-representation for wiki↔OA alignment — refuted

**Idea.** Borrow `~/src/mycooc`'s **coocmap** (the hard variant of softcoocmap): give each
corpus its own intra-cloud `V×V` association matrix, use cross-corpus NN to define
**corresponding anchor pairs**, re-represent every item by its *intra-cloud* association
profile to those anchors, recompute the cross-corpus match, and iterate. This is a
**nonlinear, non-parametric** re-representation — *not* Procrustes/VecMap (an early assessment
mischaracterized it that way; corrected). The hope: comparing *relational* profiles (how item
i relates to anchors in its **own** cloud) rather than absolute positions could recover
wiki↔OA correspondences **despite** the near-disjoint local support (register residual), *if*
the two clouds share **analogous internal relational structure**.

**How it actually works** (verified against `~/src/mycooc/coocmap_baseline.py`): intra-cloud
matrices `M_W` (V_W×V_W), `M_OA` (V_OA×V_OA); loop = CSLS on the cross-corpus sim →
forward/backward match → anchor index pairs `(inds1, inds2)` → `Xf = unit(M_W[:, inds1])`,
`Zf = unit(M_OA[:, inds2])`, `sim = Xf @ Zfᵀ`. Each item profiles only against its **own**
cloud's anchor columns; the cross-corpus comparison is between *corresponding-anchor profile
shapes*. Feeding it from dense GIST embeddings: per-corpus kNN-gram (top-k cosine, rest 0) →
`unit→center→unit`. Dictionary priming: skip the sorted-signature bootstrap (we share an
encoder), init `sim = X_W X_OAᵀ`.

**Correct success metric.** coocmap is a cross-corpus **correspondence/dictionary** method
(unsupervised translation) — success = correct cross-corpus *matches*, NOT "mixing" the clouds
(two corpora can stay locally disjoint while the dictionary is correct, exactly as
English/French embedding clouds never merge). *(An earlier run scored mixing — kNN same-corpus
purity → 0.5 — which is the wrong objective and produced a non-trustworthy refutation with an
inter-cloud diagnosis. This entry supersedes it; the verdict is the same, the reasoning is
replaced.)*

**What was tested.** NLP-wiki top-3000 (cosine to the OA-NLP centroid) vs OA-NLP 3000,
per-corpus GIST; intra-cloud kNN-gram (k=20); coocmap primitives verbatim; primed from
`X_W X_OAᵀ`; 10 iterations, seed 0. Verified prototype (code + matrix shapes shown):
`artifacts_scripts/wiki/coocmap_correspondence_retest.py` (`M_W = M_OA = (3000,3000)`,
`sim = (3000,3000)`).

**What happened — refuted (correctly).**
- **Mutual-NN collapses, doesn't climb:** raw CSLS'd prime MNN **0.373** → coocmap iter-1
  **0.134**, iter-10 0.13–0.22. On genuinely alignable data coocmap's MNN climbs *past* the
  init; here it halves on the first iteration and asymptotes far below. It *degrades* the
  dictionary.
- **Top-20-confidence correspondence precision:** raw cosine **~0.90** (exact concept matches:
  Eurotra→Eurotra, Cantonese→Cantonese, HPSG→HPSG, Code-switching→Code-Switching,
  LSA→LSA-in-MT) vs coocmap **~0.15–0.40** (mostly off; hub collapse — one OA target recurs 3×
  in the top-20).
- **98.3% of matches change vs raw, net-destructive** (precision and MNN both fall).

**Why (the diagnosis — the artifact).** coocmap wins when you have (a) *no usable starting
dictionary* but (b) *analogous internal relational structure* in both clouds. For wiki↔OA,
**(a) is already solved** — the shared encoder gives a strong cross-corpus dictionary for free
(raw prime ~0.90 precision), so there is no cold-start for coocmap to fix. And **(b) fails at
the fine grain that matters**: a concept's intra-*wiki* kNN-association profile does not mirror
its intra-*OA* profile — encyclopedic neighbors ≠ academic-citation/abstract neighbors (the
validated register residual: 41° PC-subspace angle, kNN purity 0.92). So replacing the strong
absolute-cosine signal with *mismatched relational profiles* destroys information rather than
adding it. This is established by a genuine **correspondence** failure (matches worsen + MNN
collapses), not the prior run's mixing argument.

**Salvage (and a positive finding).**
- **The raw shared-encoder cosine NN IS the usable cross-corpus dictionary** (~0.90
  top-confidence precision). That — not coocmap — is the anchor source to feed downstream
  (CSLS-mutual-NN pseudo-anchors → CORAL/CCA, if a transfer need re-motivates second-moment
  alignment). The shared encoder makes the *matching* easy; coocmap only helps the *hard*
  matching problem we don't have.
- **The live pedagogical direction is wiki-native** LPP on the link graph
  (`docs/wikipedia-pedagogical-lpp.md`) — no cross-corpus alignment needed.
- **CSLS** stays useful as an opt-in NN re-ranking knob (already ported / backlogged).
- The earlier "this bounds CORAL/CCA too" claim is **withdrawn** — it rested on the
  wrong-metric run. CORAL/CCA remain untested; they now have a strong anchor source.

**Pointers.**
- Verified prototype: `artifacts_scripts/wiki/coocmap_correspondence_retest.py`.
- coocmap algorithm: `~/src/mycooc/coocmap_baseline.py`.
- Residual characterization: `docs/wikipedia-openalex-shared-space.md`.
- Design context + tracking: `docs/backlog/curriculum-generalization-pedagogical-subspace.md`.
