# Fix curriculum specificity: semantic re-weighting (citation-central ≠ on-topic)

**Problem (the dominant, universal eval defect).** Our gap-seeded curricula lose **specificity**
on every book in the ToC-reconstruction panel (both judges) — they wander into off-topic
*foundations* (networking, databases, generic ML, signal processing, linear algebra). Diagnosis:
those foundations are **citation-central** (heavily cited by the gap's seed articles and central
in the field's citation graph) but **semantically distant** from the gap topic. Pure-citation
PageRank ranking therefore promotes them; the boundary selector's field-local background
(`p^a·log(p/g_local)`) demotes field-central works but the foundations are *both* field-central
*and* cited-by-the-gap, so they survive. The missing signal is **semantic distance from the gap**.

Key nuance from the eval: the book's *own* ToC includes prerequisites too (e.g. the Mobile
Crowdsensing book has "Mobile Sensing", "Mobile Crowdsourcing" chapters) — but they're
**topic-proximal** prerequisites, not generic-field foundations. opus rewards coherent
prerequisites (progressive); gemma penalizes any. So the goal isn't "remove all prerequisites"
— it's "keep topic-proximal ones, drop generic-field ones," which is exactly what
**semantic-similarity-to-the-gap** distinguishes (topic-proximal = close; generic = far).

**Ideas, cheapest first:**
1. **Semantic re-weight of selection (recommended first).** After ranking, blend the relevance
   score with cosine similarity to the gap centroid: `score = relevance^a · sim(work, gap_centroid)^b`
   (or a hard post-filter at a sim threshold). Directly demotes citation-central-but-semantically-
   distant foundations while keeping topic-proximal ones. Cheap (we already have embeddings + the
   gap centroid); a knob `b` controls how aggressively. Risk: a single centroid is crude for a
   multi-cluster gap — use max-sim over the gap's seed articles (or per-cluster centroids) instead.
2. **Similarity-graph mixing (principled).** Mix an embedding-kNN graph into the citation graph
   *before* PageRank (the existing `--similarity` option) so the walk prefers semantically-similar
   neighbors. The right fix. It OOM'd only at **domain scope** (full-corpus self-kNN over 40.9M
   works); the build is scope-proportional (`knn(cloud, cloud)`), so at AI-subfield (~1.55M) and
   CS-field (~6.8M, the panel scope) it fits in ~5 / ~25 GB — no scoping hack needed, just run it.
   `book_curriculum.py` needs a `--similarity` flag added (mirror `--boundary`; `seed_from_gap`
   already has it). Directly testable on the panel — arguably the better first experiment than (1)
   since it changes the walk itself rather than post-hoc re-weighting.
3. **Two-section structure (presentation, orthogonal + cheap).** Split the curriculum into a
   labeled "Prerequisites" section (the foundations) and "Core" (on-topic) chapters. A curriculum
   *should* have prereqs; the judge reads mixed-in foundations as off-topic *wandering*. Labeling
   them addresses the perception directly without changing selection. Pair with (1) to bound how
   many prereqs get included.
4. **Stronger background penalty / shorter walks** — raise the `g_local` exponent, or lower the
   PageRank `alpha` (more teleport-to-seeds → fewer citation hops to deep foundations). Tunable
   knobs; risk of over-demoting legitimate central topics.
5. **Data-driven per-edge / per-node semantic edge-weighting (best successor to the global mix).**
   The global `mix_weight` failed empirically (neuromorphic, w=0.1: book_toc still wins specificity
   4/4 axes; the statistical-learning + computing-history/lambda-calculi foundation clusters
   survived) because a single scalar can't distinguish an on-topic citation from an off-topic one.
   Instead let the data set the weight locally:
   - **Per-edge:** re-weight the *existing* citation edges (no separate kNN graph) by endpoint
     similarity: `A'[u→v] = A[u→v]·exp(β·(cos(e_u,e_v)−1))`. Keeps citation topology, damps
     semantically-distant jumps. Reuses `SimilarityGraphConfig.sharpness`'s `exp(sharpness·(cos−1))`
     form, applied to `A_cite` entrywise. Strictly dominates the global mix (spends budget only on
     bad edges).
   - **Per-node (coarser):** scale a node's edges by `s(v)=max_sim(e_v, gap_seeds)`. **In/out
     asymmetry is the lever:** foundations are citation-central via high *in-degree*, so damping
     *in-edges* to semantically-distant nodes directly deflates their inflated rank — the most
     targeted fix for "citation-central but off-topic." = topic-sensitive PageRank on the
     *transitions*, not just the (already gap-personalized) teleport.
   - **Set `β` from data:** pick it so the half-weight cosine `1+ln(0.5)/β` lands at the median (or
     p25) of the empirical citation-endpoint cosine distribution → per-scope, no hand-tuned knob.
   - **Don't** chase a data-driven *global* `w` by maximizing coherence-to-gap: degenerate at `w→1`
     (pure-semantics ball discards the citation structure that's the whole point of using citations).

**Eval is already wired:** the ToC-reconstruction panel (`docs/curriculum-toc-reconstruction-eval.md`)
gives a direct specificity read per book; re-run the panel after each variant and watch the
specificity column (currently 0/6 for ours). Cross-ref: [[curriculum-ranking-input-levers]]
(input-side fixes), [[curriculum-reciprocal-selection-results]] (the boundary selector this builds on).

## Status / working notes (2026-06-02)

**Global similarity-mixing (approach 2) tested and REFUTED for this defect.** Added `--similarity`
to `book_curriculum.py` (mirrors `seed_from_gap`) and ran Neuromorphic Photonics at CS-field
(field=17, boundary + similarity, 13 chapters to match the ToC), judged gemma4:31b pairwise vs the
unchanged Track B (`--n-titles 0`, repeats=3, 0 parse failures):

| mix_weight | interestingness | comprehensiveness | specificity | progressive |
|---|---|---|---|---|
| 0.1 | book_toc | book_toc | **book_toc** | book_toc |
| 0.5 | book_toc | book_toc | **book_toc** | book_toc |

book_toc wins all four axes, **position-robust** (both presentation orders agree after
de-randomization). The two off-topic clusters survive *both* weights: a generic-ML chapter
(`learning, theory, statistical, machine learning`) and a general-computing chapter
(`computing, systems, memory, computation, algorithms`, with lambda-calculi/1960s-computing in the
tail). gemma's specificity verdict pins the loss exactly there ("several early chapters on general
neural network theory and computer science basics"). **Diagnosis:** the convex blend *adds*
semantic kNN edges but doesn't *subtract* the foundations' citation centrality (high in-degree), so
they stay in the top-5000 ranking → still form chapters. A single global scalar is the wrong
instrument — confirming the move to **approach 5 (per-edge / per-node data-driven weighting)**,
which attenuates the bad citation edges rather than diluting all of them uniformly.

**Compute note (relevant to running the panel with `--similarity`):** `get_cloud_index` is already
`@MEMORY.cache`'d over the `faiss.Index`. The cold build over the 6.8M CS-field cloud took **634 s**
(efc=384) and persisted an 11.5 GB pickle; the second run *loaded* it and
`build_similarity_adjacency_from_cloud` dropped from ~10.5 min to **10.5 s**. So a mix_weight (or
per-edge `β`) sweep is cheap after the first build, and the panel's first `--similarity` book warms
the index for the rest at that scope. Domain-scope `--similarity` still OOMs (self-kNN over 40.9M).
