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
1. **Product-of-experts selection (recommended; the "what a statistician does" version).** Citation
   importance and semantic proximity are two *independent* relevance signals, so combine them by a
   product (rank by their product = additive in log space) — no graph surgery, no tuned `β`:

       log score(v) = log π(v) − log π_null(v) + log f̂_seeds(v)
                      └── excess citation centrality ──┘  └ semantic density ┘

   - **Semantic term = a kernel density estimate, not an ad-hoc cosine kernel.** The gap seeds are a
     sample; `f̂(v)` is their Parzen-window density at `v`. On unit embeddings the Gaussian kernel is
     `exp((cos(v,s)−1)/h²)`, so `log f̂(v) = logsumexp_s[(cos−1)/h²] − log n`. Bandwidth `h²` = median
     pairwise squared distance among seeds (**median heuristic** — data-driven, no knob). This
     naturally generalizes "max-sim to seeds" into a proper density. Demotes foundations (low seed
     density) while keeping topic-proximal prereqs (high density).
   - **Citation term = excess over a popularity null.** PageRank ≈ degree, so use `log(π/π_null)`
     (observed-over-expected, à la PMI/IDF) to keep only centrality beyond what popularity explains.
     Reuse the existing field-local `g_local` (`boundary_background`) as `π_null` — a better,
     topic-aware null than global uniform.
   - Keeps PageRank intact (a well-understood estimator) and multiplies by an independently-estimated
     semantic likelihood; the three log-terms are interpretable and auditable. **Don't** chase a
     `cos^b` exponent or a per-edge `exp(β(cos−1))` kernel — those are the parts a colleague would
     call ad hoc. Implemented as `cfg.semantic_rerank` / `book_curriculum --semantic-rerank` (see
     working notes for the result).
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

**Product-of-experts re-rank (approach 1) tested — fixes the target defect.** Implemented
`cfg.semantic_rerank` / `book_curriculum --semantic-rerank`: `log(p/g_local) + log f̂(seed KDE)`,
bandwidth = median heuristic (h²=0.25 here), KDE over 6.8M works in ~63 s, no HNSW needed.
Neuromorphic, CS-field, 13 chapters, gemma4:31b pairwise vs Track B:

- **Foundation-wandering is gone.** Both off-topic foundation chapters (statistical-learning,
  general-computing/lambda-calculi) were evicted — all 13 chapters are now photonic/neuromorphic,
  and gemma's specificity complaint shifted *entirely off* foundations. This worked even though the
  citation term out-scales the semantic term ~13:1 (cite span ~25 nats, sem span ~1.9 nats) — a
  ~1.9-nat penalty was enough at the top-5000 margin.
- **Verdict didn't flip** (book_toc still wins 4/4) because a *separate* defect now dominates
  specificity: a **debris chapter** (ch3 = a real "polarized optical processing / computational
  paradigms" cluster that absorbed a few noise works → phrase-mining surfaced junk tokens `tesla`,
  `santa`, `crayfish`, `rechenfehlern`, `proposte`). gemma robustly flags it in both orders. This
  cluster appears in the baseline and similarity runs too — foundations were just louder before.
- comprehensiveness/interestingness comments now often *favor* ours ("specific hardware … VCSELs
  and microring weight banks … more compelling for a specialist reader"), losing narrowly, dragged
  by the same debris chapter.

**Net:** approach 1 solves foundation-wandering cleanly; the residual loss is the orthogonal
**debris** problem (phrase-mining/cluster noise), which is the next thing to attack — plus
re-running the other panel books to confirm the foundation fix generalizes.

**Compute note (relevant to running the panel with `--similarity`):** `get_cloud_index` is already
`@MEMORY.cache`'d over the `faiss.Index`. The cold build over the 6.8M CS-field cloud took **634 s**
(efc=384) and persisted an 11.5 GB pickle; the second run *loaded* it and
`build_similarity_adjacency_from_cloud` dropped from ~10.5 min to **10.5 s**. So a mix_weight (or
per-edge `β`) sweep is cheap after the first build, and the panel's first `--similarity` book warms
the index for the rest at that scope. Domain-scope `--similarity` still OOMs (self-kNN over 40.9M).
