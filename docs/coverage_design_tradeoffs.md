# Coverage-detection design space: tradeoffs between winners

Successor to `docs/cohesion_design_axes.md`. The original doc captures
the per-gap *quality metric* design (kernel form, boundary
construction, ranking score, etc.) — most of those choices stand. This
doc captures **new axes that emerged from the Phase 1-3 implementation
and the soft-EM prototype**, with empirical comparisons between
winners-so-far. The original axes are referenced where relevant but
not re-litigated.

Read this alongside:
- `docs/cohesion_design_axes.md` — original 14-axis investigation
- `docs/cohesion_aware_gap_quality.md` — Phase 1-3 design as shipped
- `docs/gap_method_comparison.md` — top-N comparison across pipelines

---

## What's new since `cohesion_design_axes.md`

That doc treats the gap as a single hard-set basin discovered by one
expansion algorithm. The recent work surfaces three orthogonal design
dimensions it doesn't cover:

1. **Inner-loop policy beyond strict-vs-lenient.** The original §D
   identified strict and lenient as the competitive policies. We've
   since explored batch-independent (refuted), ordered cumulative
   (current Phase 2 winner), and soft-EM (prototyped today). Each has
   distinct confluence/resolution profiles.
2. **Books as participants, not just boundary anchors.** Original
   treated books only as boundary objects. Phase 3 makes them
   **seeds** and (under `--with-book-seeds`) potential **G members**.
   Each is a binary axis.
3. **Coverage labeling separate from basin discovery.** Phase 1's
   vote-concentration heuristic, Phase 3's structural
   `book_anchored = (n_books > 0)`, and (today's prototype) multi-
   membership inheritance are three different ways to answer "is this
   gap covered?" — and they disagree productively.

Plus a meta-observation: **what counts as "the cluster" is itself a
design choice.** The cohesion metric naturally surfaces *tight semantic
specialties*; K-ball naturally produces *broader textbook-territory
basins*. Same metric, different granularities depending on stopping
rule. This affects whether a given basin is "covered" — the same
articles can simultaneously be *uncovered* in their tight specialty
and *covered* in a broader containing basin.

---

## Axis I: Inner-loop expansion policy

How does the basin grow at each iteration?

| Variant | Per-iter cost | Confluence (T11303 disable-fast-filters) | Top cluster size (T11303 prod) | Notes |
|---|---|---|---|---|
| **Strict-single-add** (original §D winner) | 1 candidate eval per outer iter | 56.9% | 80–150 articles | Greedy, hard, slow per seed. Stable, well-understood. |
| **Batch-independent ADD** | W candidate evals per outer iter, all-or-nothing | 35.7% | ~2,400 articles | **Refuted.** Ignores collective effect on `min_post`; produces runaway clusters. |
| **Ordered cumulative ADD + symmetric REMOVE** (current) | W candidate evals, evaluated against running G | 48.8% | 70–172 articles | Phase 2 shipped. Strict-like cluster sizes, more confluent than strict, faster (mean=3.4 iters vs 25.5). |
| **Soft EM (fuzzy weights)** (prototype) | W posterior updates per iter, no thresholds | 14 iters / 3.4s for one T10181 seed | 89 articles (cross-lingual UD example) | Smooth dynamics, no Viterbi artifacts. Per-basin runs faster than ordered cumulative. Discovery-pass infrastructure not yet adapted. |

**Tradeoffs:**

- **Strict** vs **ordered cumulative**: ordered cumulative has comparable cluster sizes and modestly improved confluence at much lower wall-time per seed. ordered cumulative is the right default for hard-membership pipelines.
- **Hard membership** (strict / ordered cumulative) vs **soft EM**: soft EM eliminates the Viterbi artifacts (threshold brittleness, frozenset-equality confluence noise) at the cost of a different output shape (sparse weight vector instead of frozenset). Soft EM also reveals "shadow members" — points just below τ that are structurally close to the basin — which the hard variants discard.
- **Strict** vs **batch-independent**: settled — batch-independent is refuted, see `docs/dead_ends/...` (eventual home if this gets a write-up).

**Open question:** does soft EM scale to whole-corpus discovery (88K seeds × per-seed convergence) within the same wall-time budget as ordered cumulative? Single-seed timing (3.4s on T10181) suggests yes if we don't double the convergence iter count, but the constant factor of the dense weight-update step is unmeasured.

---

## Axis II: Membership type

| Variant | Output shape | Multi-scale? | Notes |
|---|---|---|---|
| **Hard set** (current) | `frozenset[int]` | No, exclusive within ~50% Jaccard via claimed-set | Simple, fast, supports exact-match dedup. |
| **Fuzzy weights** (prototype) | `np.ndarray[float]` of length n | Per-basin; an article *can* have weight in multiple basins if we run multiple seeds | Soft EM output. Smooth but doesn't directly enable cross-basin membership inheritance. |
| **Multi-scale (post-hoc)** (prototype) | `dict[basin_id, posterior]` per article | Explicit | Recovers multi-scale membership by re-evaluating each article's `compute_state` posterior under each basin's structure. Coverage labeling emerges naturally. |

The two prototypes are stackable: fuzzy weights for *discovery* (better convergence, no Viterbi artifacts), multi-scale post-hoc for *coverage labeling* (cross-basin inheritance).

**Empirical signal for multi-scale:** the cross-lingual UD parsing
basin (89 articles) has 0% hard intersection with the broad rank-4
Parsing basin's frozenset, but **89/89 articles have posterior 0.61–0.68
under rank-4's `compute_state`**. Under multi-scale labeling, all 89
inherit book-anchored coverage from rank-4 (which has 64 books).

This is the resolution to the "Phase 3 missed the Dependency Parsing
book" concern: at the metric's actual semantics, all 89 narrow-basin
articles ARE structural members of the broader book-anchored basin.
The "miss" was a discovery-order artifact (claimed-set early-abort),
not a true structural fact.

---

## Axis III: Book role

Three independent binary choices:

| Sub-axis | Off | On | Phase 3 default |
|---|---|---|---|
| **Books as boundary anchors** | n/a | always | always-on |
| **Books as seeds** | only article seeds | book seeds run before article seeds | on (`--with-book-seeds`) |
| **Books as G members** | article-only G; frontier filtered to articles | books may enter G through expansion; frontier unfiltered | on (gated by `--with-book-seeds`) |

**Tradeoffs:**

- **Boundary anchors only** is the original `cohesion_design_axes.md` model. Books exist *outside* G, contribute to `K_B`, and pull articles toward boundary status. Simple and interpretable.
- **Books as seeds** discovers basins centered on books, which is necessary if we want every textbook-territory cluster discovered (not just those an article seed happens to reach). Costs: book seeds with empty `claimed` are expensive (the T10181 first-1000-seeds-take-12-min issue).
- **Books as members** is required for the basin's `book_anchored` flag to be structural. If books can't enter G, `n_books > 0` is impossible and we revert to heuristic-only coverage detection.

**Phase 3 takes all three on; soft-EM prototype experiments suggest the
second and third may be redundant** — book-seeded basins (Experiment 1)
produce singletons where the book itself drops below support threshold,
because the book's posterior under its own-article basin is low (the
book is on the *boundary* of articles, not at their centroid). The
"book as seed → book as anchor" pathway is weaker than expected.

**Implication:** for soft-EM-based coverage detection, we may want
**books as members but NOT as seeds** — articles do the discovery,
books are pulled in via the unconditional frontier when geometrically
close, and `book_anchored` falls out naturally. This is simpler than
Phase 3 and handles the "book on the boundary, not the centroid" case
correctly because soft EM gives the boundary book moderate (not zero)
weight rather than discarding it.

---

## Axis IV: Coverage labeling

How is "is this gap covered by a textbook?" computed?

| Method | Input | Output | Sensitivity profile |
|---|---|---|---|
| **Vote-concentration heuristic** (Phase 1) | gap members, scope embeddings, top-1-NN-book votes, book title | `book_exists_flag = (vote_concentration ≥ 0.25 AND title_keyword_overlap ≥ 0.20)` | Catches "obvious title match" cases (Dependency Parsing → "Dependency Parsing"); misses tangential book titles or diffuse vote distributions. T10181: 18/100 top-100 basins flagged. |
| **Structural anchoring** (Phase 3) | basin's hard frozenset, member types | `book_anchored = (n_books_in_G > 0)` | Catches anything where the book is geometrically close enough to be pulled in by expansion. Brittle to discovery-order claiming (Tasks 5-9 evolution). T10181: 111/100+ top-100 basins flagged. |
| **Multi-scale inheritance (post-hoc)** (prototype) | basins + per-(basin, article) posteriors | `coverage(article) = max over basins-containing-article of basin.book_anchored, weighted by posterior` | Catches cases where the narrow basin doesn't have the book but is structurally inside a broader book-anchored basin. Recovers from discovery-order claiming. Untested at scale. |

**The three are not redundant; they're a sensitivity hierarchy.**

- **Vote-concentration**: high specificity (rare false positives), low recall (~18% of basins it could flag).
- **Structural anchoring**: medium-high specificity, medium recall. Brittle to expansion-loop choices.
- **Multi-scale inheritance**: high recall (89/89 cross-lingual UD articles get covered via rank-4); specificity depends on threshold (posterior ≥ τ_inherit).

**Recommended composite:** `coverage_label =`
- `templated-leak` if Phase 1 templated_leak_flag fires
- `book-anchored` if Phase 3 `n_books > 0` (structural)
- `book-exists` if Phase 1 vote-concentration heuristic fires
- `inherited-coverage` if multi-scale post-hoc finds a containing book-anchored basin
- `real-gap` otherwise

The five-label hierarchy preserves more signal than the current four.

---

## Axis V: Cluster granularity

Same metric, same data, different stopping rules → different cluster
shapes. **This is the meta-axis the past few days revealed.**

| Stopping rule | Cluster shape | Coverage implication |
|---|---|---|
| **K-ball** (radius_fraction × dist-to-1-NN-book) | Heterogeneous (mixes multiple subspecialties under a coarse label) | "Covered" if any subspecialty has a textbook |
| **Cohesion-aware** (`min_post ≥ τ`, hard or soft) | Tight semantic specialty | Specifically covered iff narrow-specialty book exists |
| **Multi-scale** (cohesion at multiple τ levels, or multi-membership) | Both, simultaneously | Coverage at each scale separately |

**The Dependency Parsing example illustrates:**
- K-ball #10 (166 articles, "Dependency Parsing"): broad textbook-territory cluster. Mixes cross-lingual transfer + classic methodology + multilingual treebanks. Tagged covered (via "Dependency Parsing" textbook).
- Phase 3 narrow basin (89 articles, "cross-lingual UD parsing"): tight semantic specialty. *Specifically* uncovered (no specific textbook). *Broadly* covered (member of rank-4 mega-cluster which has the DP book).

Both labels are valid. The granularity choice determines which we surface to a reader.

**Open design question:** for the proposal-generation downstream (the original goal — generate textbook proposals for uncovered subspecialties), is the right input the *specifically uncovered* tight basin (89-article cross-lingual UD parsing), or the *broadly uncovered* set (basins with no inheritance from book-anchored containers)? Probably the former — but this needs explicit thinking.

---

## Empirical winners summary (T11303 + T10181 evidence)

| Decision | Current shipped | Prototype candidate | Defer |
|---|---|---|---|
| Inner-loop policy | Ordered cumulative ADD + REMOVE | Soft EM (faster convergence, smoother) | Strict (slower, no clear win) |
| Membership type | Hard set | Fuzzy weights for discovery, multi-scale dict for labeling | — |
| Book role: boundary | On | On | — |
| Book role: seeds | On | Off (book-seeded basins are weak) | — |
| Book role: members | On (if seeds on) | On regardless of seeds | — |
| Coverage labeling | 4-label (Phase 1+3) | 5-label (+ inherited-coverage) | — |
| Cluster granularity | Cohesion-aware tight | Multi-scale (tight + broad simultaneously) | K-ball-only |

---

## Concrete next-step candidates

In rough order of cost/value:

1. **Multi-scale post-hoc labeling** (low cost, high value). Run the
   existing Phase 3 output's basins through pairwise posterior
   evaluation: for each (article, book-anchored basin) pair, compute
   the article's posterior under that basin. Output: enriched CSV with
   `inherited_coverage_basin_id`, `inherited_coverage_posterior`.
   Doesn't change discovery; just adds a label column.

2. **Soft-EM with books-as-members-but-not-seeds** (medium cost,
   medium-high value). Replace the Phase 2 inner loop with soft EM
   (preserving local CSLS boundary). Drop book-seeding. Articles do
   discovery; books pulled in as members via unconditional frontier;
   `book_anchored` falls out structurally.

3. **Five-label coverage hierarchy** (low cost, low-medium value).
   Update the comparison-table generator to produce the
   five-label classification. Trivial; depends on (1) for the
   `inherited-coverage` label.

4. **Lockstep / global EM** (high cost, theoretical interest). The
   N-component EM with K-NN sparsification. Whole separate project.

(2) is the natural successor to today's prototype. (1) is the cheapest
fix that addresses the "Phase 3 missed coverage" issue with concrete
evidence we ran today.
