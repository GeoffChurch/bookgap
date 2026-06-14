# End-to-end overview: from a field of knowledge to a book or a proof

*For a reader new to this codebase. It explains **what the whole project is doing** and **what we've
learned**, in plain terms, with pointers to the detailed docs for each piece. It does not assume you
know the code.*

## The one idea

Two questions turn out to have the **same shape**:

1. **Books.** Given a research field (its papers and textbooks), *where* should a new
   textbook/curriculum go, *what* should it contain, *in what order*, and *is it any good*?
2. **Proofs.** Given a body of formal mathematics (Lean's Mathlib), *which* theorems are worth
   proving, *what context* helps a prover, and *does it actually help*?

Both are instances of a single **five-stage pipeline**:

> **substrate → gap detection → selection → ordering → generation → evaluation**

We've instantiated this pipeline in **two domains** — research **articles** (an embedding +
citation space) and Lean **theorems** (a dependency graph). Most stages are explored in both; one
cell — *gap detection on theorems* — is still open. This document walks the five stages, showing
each domain side by side and the headline finding, then maps what's done vs open.

The coverage at a glance (✓✓ deep, ✓ done, — open):

| stage | articles / research | lemmas / theorems |
|---|---|---|
| 0. substrate | GIST embeddings + OpenAlex citation graph | Mathlib dependency graph |
| 1. gap detection | ✓✓ | ✓ static (validated); temporal = negative (both domains) |
| 2. selection | ✓ | ✓ |
| 3. ordering | ✓ | ✓ |
| 4. generation | ✓ | ✓ |
| 5. evaluation | ✓ | ✓ (the gold standard) |

---

## Stage 0 — Substrate: where "near" lives

Everything downstream is about distance and adjacency, so first you need a space.

- **Articles.** Each work (paper or book) is a point: a 384-d **GIST-small** text embedding, plus
  the **OpenAlex citation graph**. "Near" = semantically similar / citation-adjacent. Books and
  articles live in the *same* space, which is what lets us ask "where do books sit relative to the
  literature?" Scale (one subfield, AI): ~14,600 books, ~1.55M articles.
- **Theorems.** The **Mathlib declaration dependency graph** (~466K declarations, ~12.6M edges; an
  edge `A→B` means "A's elaborated proof term uses constant B"). "Near" = dependency adjacency.
  *Granularity note:* edges are **named declarations** (theorems/defs/instances/axioms) as used by
  the *elaborated* proof term — already richer than the source-visible premises (it includes
  typeclass/instance plumbing), but **not** expanded to primitive kernel inference steps (those
  would be orders of magnitude larger and mostly noise; the higher-signal richer direction is
  tactic-level traces or edge-weighting by premise surprisal, not kernel primitives).
  → `docs/lean_decl_graph.md`, `docs/lean-curriculum-experiment.md`.

---

## Stage 1 — Gap detection: where should the next thing go?

**The question:** where is a field *under-served* — the place a new book (or theorem) belongs?

**Articles (deeply explored).**
- A book's **own-gap** = the ball of articles around it out to its nearest *other* book
  (`book_own_gap.py`). The clean structural result: **comprehensive textbooks have own-gap ≈ 0**
  (they sit in book-dense regions — Jurafsky, Koehn, AIMA all ≈0), while **niche books fill the
  real gaps** (only the top ~1% of books have a usable own-gap). The two virtues are
  *anticorrelated*: niche ⇒ real gap but usually a monograph; canonical ⇒ clean ToC but no gap.
- There is a **zoo of ~14 gap definitions** and head-to-head **bake-offs** (no single winner — it's
  regime- and *species*-dependent: radial gaps → density-ratio, concept gaps → a keyword-lattice
  miner, corridor gaps → a voltage/current detector).
- **What makes human books *maximally exceptional*?** (this session) Scoring every definition as a
  point-test of books vs a matched null: the winner is the **density log-ratio
  `Δ = log r_F,k − log r_B,k`** (article-distance minus book-distance) vs a matched-control null,
  clustered bootstrap. **Robust at scale:** A3 stays the book-level winner at **AUC 0.691 [0.674,
  0.710]** over 29 books / 1,717 groups (was 0.738 on 4 niche books). But CI-tightening *corrected*
  the **chapter-level** story: kde's `0.725 [0.657, 0.789]` (47 chapters) fell to **0.564 [0.538,
  0.596]** at 371 chapters — the small-sample number was optimism; chapters are only *modestly*
  exceptional. (And the unit-inversion is sharp: the book winner A3 scores 0.43 at chapter level,
  the chapter winner kde scores 0.46 at book level — the two units want opposite definitions.)
  Strikingly, **raw
  emptiness *anti*-correlates** — books sit in article-*dense* pockets, so it's the *ratio*
  (articles far yet books near), not the void, that marks where books go.
  → `artifacts_scripts/curriculum/book_surprise_bakeoff.py`.
- **Chapters fill gaps even when the book doesn't** — decomposing a canonical textbook into
  chapters recovers per-chapter gaps the book-level metric reports as zero (the workaround for the
  canonical-book limitation). → `docs/curriculum-toc-dendrogram-matching.md`.
- *Honest caveat:* predicting *where human effort will go* from a single snapshot is mostly a
  **negative** result — citation-flow "reroute" (the backbone the field routes through) and
  arrival-density (where new papers land) are *orthogonal*. → `docs/gap-detection-worklog.md`,
  `docs/gap_method_comparison.md`, `docs/backlog/curriculum-applications.md`,
  `artifacts/gap_detection_eval/report.md`.

**Theorems (open cell — now *validated by plant-and-recover*; a clean domain inversion).** The
analogy ports (theorems = F, defs = B; ~190K vs ~38K math-core decls), and a quick prototype's
*ranking* is — as in the article domain — dominated by **definitional scaffolding** (`zpowRec`,
`Function.Injective.*`: the Lean "templated-leak"). But the decisive test — **plant-and-recover**
(delete a concept's theorem cluster, can a detector recover the orphaned region?), null-calibrated
against *local* deletions, two seeds, clustered bootstrap — gives a surprising, validated verdict:
the **graph citation-ratio** `Δ = log(def_users+1) − log(thm_users+1)`, recomputed against the
post-deletion graph, is the **only** detector with real signal (lift **+0.17 / +0.13** over the
local null, CIs excluding 0; 35/40 plants recovered), while the **embedding density-ratio and
level-set *regions* — the article domain's winners — FAIL** (lift ≤0; level-set flat at 0.50, recall
0). The mechanism is unifying: **Lean's dependency graph is a *clean oracle*** (proof-term `uses`
edges), so a citation-count signal that was noisy/circular for Wikipedia is *sharp* here, whereas
bge-small embeddings of symbol-heavy type statements place unrelated objects together and let
scaffolding dominate. **So the right gap detector flips by domain** — graph-citation in Lean,
embedding density-ratio in articles. **But a second result cuts the other way:** the git-tag
temporal holdout is *negative* — M1 recovers *planted* concepts (static) yet does **not predict
where new theorems land** (centrality-matched AUC 0.50; what actually attracts new theorems is
*preferential attachment* — already-theorized defs grow ~2.4× their matched null — the **opposite**
of a gap). This mirrors the article temporal result exactly, so the *temporal* finding is
**universal: a gap does not forecast arrivals — rich-get-richer does**, and the clean oracle did
**not** rescue prediction.

---

## Stage 2 — Selection: what to include?

**The question:** which items belong in the curriculum/book/chapter (or the prover's context)?

- **Articles.** Rank by **target-seeded PageRank** with a **background-normalized** score
  (`p^a · log((p+ε)/(g+ε))`) → flat top-N; or cut a **dendrogram** of the field into topic clusters
  (the `effN_sil` adaptive cut, etc.). A key finding: under soft optimal-transport matching, the
  cut *method* barely matters — clustering is second-order. → `docs/wikipedia-curriculum-bgnorm.md`,
  `docs/curriculum-toc-dendrogram-matching.md`.
- **Theorems.** `t1` = curriculum-selected premise statements (dependency-graph centrality, flat
  top-N, filtered to theorem-kind). The pre-registered verdict: **`t1` *ties* BM25** — graph
  "curriculum" selection does **not** beat equal-budget lexical retrieval. What helps a prover is
  *relevant premise retrieval*, not curriculum specifically; every elaboration (bidirectional
  PageRank, more depth, diversification) **hurt or was null**. → `docs/lean-confirmatory-results.md`,
  `docs/lean-curriculum-experiment.md`.

**Cross-cutting lesson:** *"relevant" is the win.* Machinery beyond "give me the relevant items"
keeps failing to pay — in both domains.

---

## Stage 3 — Ordering: in what sequence?

**The question:** given the selected items, what order?

- **Articles.** Human textbook chapters are **not** a global sweep of the topic structure
  (Spearman of chapter-number vs dendrogram order is null), **but** they follow **good local
  edges** — consecutive chapters are topically adjacent far more than chance. And this is a
  **genre signal**: a single-author *textbook* orders this way (Jurafsky 98–100th percentile vs
  random), an *edited volume* doesn't (random, even at high corpus-fit). A greedy good-edge
  generator *recovers* the human order (Kendall-τ **+0.39, p=0.008**, n=25 chapters; the per-book
  good-edge percentiles are from single books and would tighten with more ToCs). → `docs/curriculum-toc-dendrogram-matching.md`.
- **Theorems.** `t1` vs `t1_shuffled` (same premises, order randomized): **ordering is inert** — the
  pre-registered H2 was *falsified*. Selection, not order, carries the whole effect.
  → `docs/lean-confirmatory-results.md`.

**Cross-cutting lesson:** ordering matters for *human-authored textbooks* (and is genre-bound), but
**not** for a prover's context window.

---

## Stage 4 — Generation: produce the artifact

**The question:** turn the gap/curriculum into a real thing — a compiled book, or a closed proof.

- **Articles → a textbook.** The `learn-and-teach` textbook compiler turns **one opaque "brief"**
  into an autonomous LaTeX book via a frozen pipeline (ToC tournament → subagent drafters → a
  `book_digest` acceptance oracle). The data-science ablation arc is instructive: r1 solo-collapsed,
  r2 wrote all chapters but wouldn't assemble, r3 forced per-chapter compilation → a 140pp book
  passing 7/8 gates, r4's overfull-forcing *overshot* a weak model's iteration ceiling and
  regressed. Recurring failure modes: weak models **evade gates** (the "gate arms race") and hit a
  **long-context degeneration ceiling**. → `~/src/learn-and-teach/compilers/textbook` (the build
  logs live under its `outputs/`).
- **Theorems → a proof.** A prover (Goedel-Prover-32B; gemma-31B as the weak secondary) is given
  the selected/ordered context and tries to close the goal. → `docs/lean-eval-harness.md`,
  `docs/goedel-prover-serving.md`.

**Cross-cutting lesson:** relevant context lifts the *strong* prover ~6× — but it's
**capability-gated** (the weak prover, near the floor, barely benefits), and weak *generators* hit
a context ceiling in both domains (book degeneration; gemma's near-zero proving).

---

## Stage 5 — Evaluation: how do we know it's any good?

The hardest stage, and where most of the machinery lives.

- **Articles.** Three families, each with a catch: **LLM-as-judge** (and its failure — a blind judge
  ranks the most field-comprehensive ToC *last*, and judge-vs-metric verdicts dissociate almost
  completely); **temporal holdout** (predict gaps at time T, score against the papers that arrive by
  T+Δ — mostly a *negative* result; only level-set *change* survives); **book-rediscovery recall**
  (hide 20% of real books, do the method's top gaps recover them?). The cleanest static test is the
  **book-surprise AUC** (books vs a matched-control null, clustered bootstrap). → `artifacts/gap_detection_eval/`,
  `docs/curriculum-toc-reconstruction-eval.md`, the Showdown readouts.
- **Theorems (the gold standard).** The Lean **kernel is a ground-truth oracle**: a proof either
  checks or it doesn't. The metric is **audited pass@3** — the verifier accepts *and* a
  constant-dependency audit confirms the proof didn't just cite the withheld target (circular
  "successes" are demoted, never pooled). Everything was **pre-registered and signed** before the
  run; comparisons are exact McNemar. Verdicts: H1 not supported (curriculum ties retrieval), H2
  falsified (order inert), H3 supported (content not warmth), H4 supported (capability-relativity),
  H5 supported (harder = higher premise surprisal). → `docs/preregistration-curriculum-context-v2.md`,
  `docs/lean-confirmatory-results.md`.

**Cross-cutting lesson — the most important in the whole project:** judges and metrics
**dissociate**, so you need a **ground-truth oracle**. The theorem side *has* one (the verifier),
which is why its results are crisp; the article side must fight judge/metric unreliability, and even
its best signals don't predict *future* human effort from a snapshot.

---

## The map, the holes, and future work

**What's solid.** Article gap detection (esp. the density-ratio / book-surprise result); the
pre-registered Lean selection/ordering verdicts; the two generators; the Lean verifier as a clean
evaluation oracle.

**The open cell — gap detection on theorems** — now has a *validated* detector, with a twist.
Plant-and-recover confirms the **graph citation-ratio** recovers planted concept-deletions
(lift **+0.17** over a local-deletion null, two seeds, CIs exclude 0), while the embedding and
level-set levers — the *article* winners — **fail**: a clean domain inversion (Lean's dep graph is a
clean oracle, so the graph signal is sharp; article citations are noisy, so they need embedding
density-ratio + regions). **Done — and negative:** the git-tag temporal holdout shows M1's static recovery does **not**
transfer to *predicting* arrivals (centrality-matched AUC 0.50; preferential attachment, not gaps,
drives where new theorems land — mirroring the article temporal negative). So the cell is
*resolved*: Lean has a validated **static** gap detector, but in **both** domains gaps don't forecast
the field's future — a universal negative, independent of oracle cleanness.

**Other open threads.**
- *Ordering:* a causal corpus-fit test (does order-recovery strengthen with a tighter per-book
  corpus?) and a centrality-based **intro-node** generator (the "cycle in the dendrogram" idea).
- *Gap prediction:* the snapshot→arrival signal is weak; differenced/change-form signals are the
  surviving lead.
- *Generation:* the weak-model context ceiling is the binding constraint; stronger executors or
  smaller per-unit context are the levers.
- *Confidence intervals (book-surprise now tightened):* re-run over **29 books / 371 chapters** —
  book-level A3 is robust (0.691 [.674,.710]), but chapter-level kde *shrank* from a small-sample
  0.725 to **0.564 [.538,.596]** (real but weak — a CI-tightening that corrected a conclusion). The
  **ToC-matching / ordering** findings (good-edges percentiles, the Kendall-τ) still rest on a handful
  of books and are the loose ones to tighten next.

## Deep-dive pointers

| topic | doc |
|---|---|
| Gap detection — methods, bake-offs, worklog | `docs/gap-detection-worklog.md`, `docs/gap_method_comparison.md`, `docs/backlog/curriculum-applications.md` |
| Gap detection — evaluation (plant-and-recover, temporal) | `artifacts/gap_detection_eval/report.md` |
| Book placement / book-surprise (which gap def) | memory `gap-book-surprise-definition`, `artifacts_scripts/curriculum/book_surprise_bakeoff.py` |
| Curriculum (articles) — selection & ToC matching | `docs/wikipedia-curriculum-bgnorm.md`, `docs/curriculum-toc-dendrogram-matching.md`, `docs/curriculum-toc-reconstruction-eval.md` |
| Lean curriculum experiment — the pre-reg verdict | `docs/preregistration-curriculum-context-v2.md`, `docs/lean-confirmatory-results.md` |
| Lean substrate & harness | `docs/lean_decl_graph.md`, `docs/lean-eval-harness.md` |
| Generation — the textbook compiler | `~/src/learn-and-teach` (compilers/textbook) |
