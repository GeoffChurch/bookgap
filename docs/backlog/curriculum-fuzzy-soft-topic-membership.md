# Fuzzy / soft topic membership (retire hard assignment)

**Motivation.** The pipeline is *soft on relevance, hard on assignment*: PageRank mass is
continuous, but each work is hard-assigned to one topic (`point_to_cluster` via hierarchical
k-means → dendrogram cut); masses only weight contributions *within* a topic. The integer
"works per topic" is an artifact of that argmax — `summary.md` already reports a fractional
**"Effective # of Works"** (participation ratio), so the honest count was never an integer.

**Idea.** Give each work a **soft membership** `r_{i,k}` (distribution over topics) instead of a
hard label. Use `mass × r_{i,k}` as the per-doc weight in the already-weight-aware keyword/summary
extraction (`count_topic_tokens` takes per-doc weights today — just let a work contribute to every
topic it partly belongs to).

**Why promising:**
- **Faithful** — works genuinely span topics; hard assignment forces a false either/or.
- **Dissolves padding debris at the root** — no forced singletons, so the degenerate ~1-work
  chapters that produced the junk-padding debris ([[curriculum-specificity-semantic-reweight]])
  can't form. Subsumes B1 (min-cluster-size) entirely; makes dynamic-k mostly unnecessary for that
  class (dynamic-k still useful as display honesty for genuinely thin topics).
- **Logical completion of the mass-weighting philosophy** already in the scorer.

**Chapters stay discrete.** N topic seeds/centroids (dendrogram cut) and the ordered chapter
sequence (frontier potentials on centroids) are unchanged. Only **work→topic membership for
characterization** (keywords/summary/exemplars) goes soft. Reading order still discrete; a bridging
work is listed under its top-responsibility chapter (or each chapter above a threshold).

**Cleanest realization here:** per-work **absorption probabilities to the N topic seeds** (a
topic-sensitive / absorbing PageRank — same family as the rest of the pipeline; "what fraction of
this work's flow is absorbed by each topic"). Alternatives: softmax over distance-to-centroid (soft
k-means / GMM responsibilities), NMF/archetypal/admixture. The absorbing-walk version reuses the
citation+similarity graph we already build.

**Caveats / costs:**
- Bigger change than dynamic-k — thread `r_{i,k}` through clustering → keywords → summary →
  exemplar selection.
- Does NOT fix boilerplate-cluster debris (coherent non-content clusters still group) — that's
  [[curriculum-filter-frontmatter-noncontent-works]] (ingest-level).
- Need a rule for exemplar listing under multiple chapters (threshold on responsibility).

**Sequencing:** pursue after the ingest-filter (cheaper, unblocks Color Image). Eval via the
ToC-reconstruction panel — watch specificity + whether degenerate chapters vanish.
