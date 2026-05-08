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
| **Soft EM (fuzzy weights)** (`gap_soft_em.py`, May 8) | W posterior updates per iter, no thresholds | T11303 full discovery: 16 min / 130 basins. T10181: blocked (~0.5s/seed). | T11303 top-3: BN 198A+14B / Causal 97A / GGM 113A | Smooth dynamics, no Viterbi artifacts. **5.5× confluence improvement on T11303** (130 vs Phase 3's 717). Slow at T10181 scale; needs frontier/iter optimization. |

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
| **Fuzzy weights** (`gap_soft_em.py`, May 8) | sparse `dict[idx, float]` (above τ_supp); also full weight array internally | Per-basin; an article *can* have weight in multiple basins across discovery seeds | Soft EM output. Smooth, no Viterbi artifacts. Post-hoc multi-scale pass picks up cross-basin inheritance. |
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

## Empirical findings from multi-scale labeling on T10181 (May 8 update)

Implemented multi-scale post-hoc labeling
(`artifacts_scripts/gap_multi_scale_coverage.py`) and ran on T10181's
4,925-basin Phase-3 output. Several findings disconfirm parts of the
original design and validate others.

### Single inheritance threshold doesn't work

The original sketch ("inherited-coverage if median_post ≥ τ_inherit")
implicitly assumed a bimodal distribution — basins either tightly
inside a book-anchored basin or clearly outside. The actual T10181
distribution is **unimodal: mean 0.70, std 0.07, min 0.53, max 0.97**.
At τ=0.5, every single non-book-anchored basin (4174/4174) inherits
coverage. At τ=0.7, ~50/50. No natural cutoff.

This forced a redesign: replaced the binary inherited-coverage label
with a **7-tier hierarchy** that treats containment as a spectrum:

| Tier | median_post range | T10181 count |
|---|---|---|
| `book-anchored` | n/a (`n_books > 0`) | 751 (15%) |
| `inherited-strong` | ≥ 0.75 | 1,056 (21%) |
| `inherited-moderate` | 0.65 ≤ x < 0.75 | 2,128 (43%) |
| `inherited-weak` | 0.55 ≤ x < 0.65 | 983 (20%) |
| `real-gap` | < 0.55 | 7 (0.1%) |

The `real-gap` count drops to single digits at any reasonable
threshold. **A binary "covered/uncovered" classification is wrong for
this corpus.** The continuous `inherited_coverage_median_post` is the
informative signal; tiers are just bucketed views of it.

### The "least-contained leaderboard" is the right output shape

For the proposal-generation use case, the question shifts from "which
basins are real-gaps?" to "which basins are least-covered?" Answer:
sort by `inherited_coverage_median_post` ascending. Top of the T10181
leaderboard:

| inh_med | size | coverage | topic |
|---|---|---|---|
| 0.61 | 260 | inherited-weak | RTE / textual entailment |
| 0.63 | 115 | inherited-weak | Historical OCR |
| 0.63 | 403 | inherited-weak | Wikipedia / DBpedia knowledge graphs |
| 0.64-0.65 | 100-300 | inherited-weak | OCR variants, NER, QA, Morphology |
| 0.66-0.72 | 150-350 | inherited-moderate | Dep Parsing, patent MT, G2P, etc. |

These are interpretable subspecialties. Several have textbook
candidates that were tangentially relevant but didn't pull tight basin
inheritance — Dep Parsing (top vote = "Dependency Parsing" textbook,
inh_med=0.68 i.e. inherited-moderate, *not* tight inheritance) is the
canonical example. The leaderboard correctly distinguishes "broadly
covered with strong containment" from "tangentially covered, the
'covering' basin barely contains it."

### What changed about the design

- **`book_anchored` binary kept as Tier 1** (when n_books > 0 directly).
  Phase 3's structural answer is still load-bearing for the strongest
  signal. Multi-scale is layered on top.
- **`inherited-coverage` split into three tiers.** Reflecting the
  unimodal distribution: continuous score, bucketed display.
- **`real-gap` tier becomes minority** (7 of 4925 on T10181). The
  original framing assumed real-gaps would be a meaningful fraction;
  empirically they're rare under any reasonable threshold. The
  meaningful "uncovered" signal is the *bottom of the leaderboard*,
  not a binary filter.
- **Seven labels, not five.** Per `coverage_label_v2`:
  `templated-leak > book-anchored > book-exists >
   inherited-strong > inherited-moderate > inherited-weak > real-gap`.
  The continuous score (`inherited_coverage_median_post`) is the
  primary signal; tiers are summary views.

### Implications for next iterations

- **The "fuzzy membership preserving boundary" backlog item still
  applies, but for a different reason than initially.** It's no longer
  primarily about fixing the basin-matching unreliability (multi-scale
  post-hoc resolved that). It's about getting smoother convergence
  dynamics + sparse-weight output that's natively multi-scale (an
  article's weight vector across all converged basins is exactly
  multi-scale membership without a separate post-hoc pass).
- **Books-as-seeds may be vestigial.** The Experiment-1 finding (book
  seed basins where the book itself drops below τ_supp) suggests
  book-seeded discovery isn't load-bearing. Articles do discovery;
  books participate as members where geometry warrants.
- **The 7 real-gap basins on T10181** are interpretable niches
  (interactive-predictive MT, historical NLP message systems
  (NOMAD/CIRCUS/PUNDIT), UIMA pipeline tooling, transport-domain NLP,
  Swedish temporal extraction). Whether these are good textbook
  proposal targets is a separate question — they're plausibly real
  research gaps, but they're also small (6-21 articles) and low-logit.
- **The DP textbook example** (top vote book matches the gap topic
  exactly, basin inh_med=0.68) shows the metric correctly distinguishes
  "the textbook covers a broad area that contains this basin tangentially"
  from "this basin sits inside the textbook's coverage zone." This is
  a correct distinction the K-ball baseline conflates.

---

## Empirical winners summary (post-soft-EM-validation update)

| Decision | Production-validated (T11303 + T10181) | Stronger candidate (T11303 only) |
|---|---|---|
| Inner-loop policy | Ordered cumulative ADD + REMOVE (`gap_article_as_seed.py --batch-addition`) | **Soft EM (`gap_soft_em.py`) — 5.5× better confluence, blocked at scale by per-seed cost** |
| Membership type | Hard set | Fuzzy weights (in `gap_soft_em.py`); multi-scale dict for labeling layered on top (`gap_multi_scale_coverage.py`) |
| Book role: boundary | On (always) | On (always) |
| Book role: seeds | On (`--with-book-seeds`) | **Off — book-seeded basins are weak (Exp 1: book drops below τ_supp in own basin)** |
| Book role: members | On (`--with-book-seeds`) | On (always; geometric pull via unconditional frontier) |
| Coverage labeling | 7-tier `coverage_label_v2` + continuous `inherited_coverage_median_post` for sortable leaderboards | Same — drop-in compatible |
| Cluster granularity | Cohesion-aware tight, with multi-scale post-hoc inheritance | Same — soft EM's basin shape is similar (tight specialty); multi-scale labeling is the same post-pass |

### Soft EM validation status (May 8)

- **T11303 (small, fast scope):** 130 distinct basins in 16 min vs. Phase 3's
  717 — **5.5× confluence improvement**, validating the Viterbi-artifact
  prediction. Top-3 basins (Bayesian Networks 198A+14B; Causal Discovery
  97A; Gaussian Graphical Models 113A) are semantically clean. Multi-scale
  labeling on the soft-EM output: 22 book-anchored / 94 inherited-strong /
  14 inherited-moderate / 0 real-gap. Post-hoc pipeline drop-in.
- **T10181 (large scope):** **blocked** at ~0.5 seeds/sec → 49h ETA at 88K
  seeds, even with 40-core vectorized matmul saturated. Killed; not yet
  validated. Per-seed cost dominated by the dense weight-update step (every
  candidate gets a weight update each iteration, vs. ordered cumulative's
  bounded acceptance frontier).

So the production choice today remains Phase 3 + multi-scale post-hoc
(validated end-to-end on T10181). Soft EM is the strongest theoretical
candidate but needs scale work before becoming the default.

---

## Concrete next-step candidates

Updated post-implementation; both (1) and (2) below are *done*; (3)
through (5) are forward-looking.

1. ~~Multi-scale post-hoc labeling~~ — DONE
   (`gap_multi_scale_coverage.py`, commit `b457c76` → `cd77350`).
   Validated on T10181 with 7-tier classification (4174 non-anchored
   basins → 7 real-gap at threshold 0.55).
2. ~~Soft EM with books-as-members-but-not-seeds~~ — DONE
   (`gap_soft_em.py`, commit `5ce230e`). Validated on T11303;
   T10181 blocked by per-seed compute. Drop-in compatible with
   multi-scale labeling.
3. **Soft EM scale work** (medium cost, high value).
   Sweep on T11303 (May 8) quantified the tradeoffs — finding inverts
   prior intuition: **seed-stride is dramatically better than smaller
   candidate sets**.

   T11303 sweep results (baseline = fw=200, iters=50, stride=1; 640s; 130 basins):

   | Variant | Wall | Basins | Top-25 size | Exact-match top-25 | Mean Jaccard recall |
   |---|---|---|---|---|---|
   | baseline | 640s | 130 | 3252 | 25/25 | 1.000 |
   | fw=100 | 551s | 181 | 1560 | 0/25 | 0.426 |
   | fw=50 | 441s | 260 | 820 | 0/25 | 0.239 |
   | fw=25 | 395s | 422 | 328 | 0/25 | 0.128 |
   | fw=50, iters=30 | 341s | 253 | 819 | 0/25 | 0.242 |
   | **stride=4** | **238s** | **100** | **3070** | **4/25** | **0.498** |

   - **Smaller frontier dramatically alters basin shape, not just discovery cost.**
     fw=100 produces 181 basins (vs 130) with average size halved and
     ZERO exact matches in top-25. The basins genuinely shift — soft EM
     with smaller frontier becomes "find the nearest semantic
     micro-cluster" rather than "find the natural broad basin." Each
     basin's gradient trajectory changes when fewer candidates are
     evaluated per iter, and basins converge prematurely to local
     hot-spots. This is fragmentation, not lossy compression.
   - **Smaller frontier doesn't even save much wall time:** fw=100 is
     only 14% faster, fw=25 only 38% faster. Boundary precompute and
     sigma calculation dominate, not per-iter candidate eval.
   - **Stride=4 wins both axes:** 2.7× speedup AND best preserved basin
     shape (4 exact matches, 0.498 Jaccard recall). Stride preserves the
     algorithm — confluence ensures 77% of baseline basins (100/130) are
     reachable from 1/4 of seeds.
   - **Lowering max-iters with `--max-iters 30` saves wall time without
     quality loss** *given the small-frontier regime*. Soft EM converges
     in <30 iters routinely; the 50-iter cap was conservative.

   **Recommended scale strategy: `--seed-stride 8` for T10181 + T13734
   (projected ~6h on T10181 vs the 49h at stride=1). Don't reduce
   frontier_width.**

   **3a. ~~Partial-lockstep frozenset cache~~ — DONE** (commit `83d11d4`).
   Cross-seed early-abort when current hard support exactly matches an
   already-discovered basin. T11303: 38% wall reduction (953s → 590s),
   100% basin-set overlap with baseline. T10181 stride=8 with cache:
   running. Pure speedup, zero quality impact.

   **3b. Adaptive K — REFUTED** (commit `dc30967`, opt-in flag).
   Tested with K_min=100 (936s, 13% basin overlap) and K_min=200
   (1836s, 55% overlap). Doubling K when |supp|>0.8K saturates fires
   often; matmul cost (∝ K·S) explodes. Changes basin shape because
   K affects the algorithm's gradient trajectory, not just the
   candidate pool. Net loss on both axes (slower AND different basins).
   Kept as opt-in flag for posterity but not recommended.

   **3c. GPU acceleration — UNDERPERFORMS at small basins**
   (commit `dc30967`, `--gpu` and `--fp32` flags). T11303 GPU FP64 was
   ~5× slower than CPU+cache; FP32 was 1.6× slower. FP32 quality is
   essentially identical to FP64 (129/130 basin overlap). On T11303
   with ~30-article basins, per-iter dispatch + CUDA launch overhead
   on ~50 small kernels dominates the actual matmul compute. May tip
   the scale on T10181 with 200-1000-article basins; untested. The
   real fix is **batched seed processing** (see (5)) which amortizes
   per-iter overhead across many seeds.

4. **Cross-basin soft membership without separate post-hoc pass.** Once
   soft EM is the default discovery, run multiple seeds and collect
   per-article weight vectors directly across all basins. Coverage
   inheritance becomes a per-article max over weights, no separate
   `compute_state` re-evaluation needed. Eliminates the post-hoc step.
5. **Batched seed processing on GPU.** Process N=32-128 seeds in
   parallel: each one's tiny matmul stacks into one big matmul,
   amortizing per-op overhead by N×. Requires reworking claimed-set /
   frozenset-cache to coordinate across parallel seeds (the
   claim-as-you-go pattern doesn't naturally batch). The lockstep-
   expansion backlog item naturally enables this. ~1 day of work.
   Likely the right move if we end up running these full-corpus
   discovery passes regularly. Once-off runs probably don't justify
   the restructure.
6. **Lockstep / global EM** (high cost, theoretical interest). The
   N-component EM with K-NN sparsification. Whole separate project;
   addresses the "boundary is local-CSLS-balanced, but other basins
   don't appear in the model" gap that fuzzy-per-basin doesn't fix.
   See
   [multicluster-softmax-membership](backlog/cohesion-metric-design/multicluster-softmax-membership.md)
   — the worked-out design for this direction, including a Phase 1
   post-hoc reassignment experiment (~2 days) that empirically tests
   whether the principled formulation produces materially different
   basins before committing to a Phase 2 lockstep restructure
   (~1–2 weeks). The design also subsumes Phase 1 (`gap_book_existence_filter`)
   and Phase 3 (`gap_multi_scale_coverage`) enrichment into the
   discovery step itself, collapsing the current 8-stage pipeline to
   3 stages.

(3a) is shipped. (5) is the natural next step if we need more
performance — gives both perf (amortized launch overhead) and
algorithmic cleanup (multi-cluster competition replaces single-cluster
EM with size-matched local boundary). (4) is a cleaner architecture
once (5) proves out. (6) is the architectural target once we want
to simplify the post-hoc enrichment pipeline; the
[multicluster-softmax-membership](backlog/cohesion-metric-design/multicluster-softmax-membership.md)
design doc lays it out.
