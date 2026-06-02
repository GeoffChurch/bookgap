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
   neighbors. The right fix, but it OOM'd at scale (full-corpus self-kNN, see dead_ends); scope the
   kNN to the ranked candidate set instead of the whole corpus.
3. **Two-section structure (presentation, orthogonal + cheap).** Split the curriculum into a
   labeled "Prerequisites" section (the foundations) and "Core" (on-topic) chapters. A curriculum
   *should* have prereqs; the judge reads mixed-in foundations as off-topic *wandering*. Labeling
   them addresses the perception directly without changing selection. Pair with (1) to bound how
   many prereqs get included.
4. **Stronger background penalty / shorter walks** — raise the `g_local` exponent, or lower the
   PageRank `alpha` (more teleport-to-seeds → fewer citation hops to deep foundations). Tunable
   knobs; risk of over-demoting legitimate central topics.

**Eval is already wired:** the ToC-reconstruction panel (`docs/curriculum-toc-reconstruction-eval.md`)
gives a direct specificity read per book; re-run the panel after each variant and watch the
specificity column (currently 0/6 for ours). Cross-ref: [[curriculum-ranking-input-levers]]
(input-side fixes), [[curriculum-reciprocal-selection-results]] (the boundary selector this builds on).
