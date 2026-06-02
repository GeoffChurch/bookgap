# Real-book ToC reconstruction eval (book-first, whole-domain, book's-own-gap)

**The design.** Manually pick real OA books (any topic) that have recoverable ToCs. For
each book, compute **its own gap** the way the main pipeline already does — treat the book
as a mere article and find its **nearest non-self neighboring book**, k-ball around it —
at the **whole Physical-Sciences-domain** scope. Seed *our* curriculum from that gap-ball,
and score it against the book's **own ToC converted to our format** as ground truth. Gap
validity is free (a real book filled it), gap selection has no confound (the book defines
its own location), and the eval is a quantitative reconstruction score, not a 1–7 vibe.

This supersedes the earlier "start from our computed gaps" framing: we do **not** use the
gap detector's chosen gaps. The book is the unit; its gap is intrinsic to where it sits.

## Why book-first / whole-domain is the right call
- **Kills the gap-selection confound.** Any OA book is a clean test case. The book's own
  gap percentile becomes a free **difficulty axis** — sample books from deep-gap to
  crowded-region.
- **Whole-domain scope is honest.** The nearest-non-self book must be found among a
  comprehensive book set; a narrow (single-topic) scope makes a book look gap-filling only
  because its real neighbors outside the topic were excluded (boundary artifact). The
  curriculum pipeline already runs at `domain=3` Physical Sciences (40.9M works), so this
  needs **no new infrastructure**. Report the percentile at *both* scopes (domain-wide =
  absolute gap size; AI-subfield = interpretable) — recomputing percentile against a book
  subset is near-free.
- **Reuses confirmed machinery.** Verified the book's-gap mechanism in code:
  `gaps.py:97-101` scores each book as a query point (`is_intra_cloud=True`);
  `kball_algorithm.py:203` `effective_k = k+1 if is_intra_cloud` drops the self-match (the
  "non-self" part); `gap_statistics.py:41` percentiles against the per-book distribution;
  both `gap_size_percentile` and `proposal_gap_size_percentile`
  (`results_export.py:112,291`) use it.

## Thin new entry point: book → its-own-gap → seed
`seed_from_gap` currently loads in-gap article ids from a precomputed gap context. Swap
that front-end for the book's own gap-ball; everything downstream is unchanged.

```python
# domain-scope FilterPair -> books (landmark) + articles (ambient) subclouds
from bookgap.cloud_index import get_kball_coverage
book_idx   = books.index_of(book_openalex_id)            # the chosen OA book
one_book   = books.subset([book_idx])                    # 1-point query cloud
cov = get_kball_coverage(ambient=articles, landmark=books, query=one_book,
                         radius_fraction=RF, k=1, is_intra_cloud=True)  # self-book dropped
_dists, idxs = cov[0]                                    # articles inside the book's ball
seed_article_ids = [articles.works()[i]["id"] for i in idxs]
# -> build_curriculum_for_gap(ids=seed_article_ids, ...) with the book held out of landmark
```
No greedy deflation (that's for ranking many gaps); we want the single raw ball around the
book. `is_intra_cloud=True` gives the nearest *other* book → radius = `RF × that dist`.

## Pipeline
1. **Curate ToC books (any topic).** Search OA technical books with recoverable ToCs
   (GitHub LaTeX: D2L/MML/PML/OpenStax; DOAB/OAPEN; LectureBank), match to the corpus by
   DOI/title. The covering-books scan (`/tmp/toc_candidates.json`) is *one* source, not the
   selector — e.g. *Deep Learning on Graphs*, *Quantum Information Theory* (92%, T10020),
   *Interpretable ML* (Molnar, free online), *Introduction to Coding Theory* are all
   corpus-present textbooks.
2. **Two-stage to control cost.** A full curriculum is a ~104 GB, ~20-min run (one-at-a-
   time). So first cheaply compute the *gap-ball* (one `get_kball_coverage` call) for many
   candidate books, then run the full curriculum only for a chosen spread across the gap-
   percentile range.
3. **Track A (ours).** `build_curriculum_for_gap` on the book's gap-ball articles, book
   held out of the landmark set (so we reconstruct, not copy). Boundary/similarity variants
   apply.
4. **Track B (reference).** Parse the ToC; per chapter, embed title+desc into our space and
   retrieve nearest corpus articles + keywords → the book's ToC in our format. Domain-
   adaptation sequencing, cheapest first: raw GIST-small → per-genre mean-center/whiten →
   synthetic-abstract paraphrase (LLM rewrites chapter as abstract) → learned Procrustes/CCA
   (reuses [[curriculum-generalization-pedagogical-subspace]]).
5. **Eval — reconstruction quality (the payoff).** Track A vs Track B, quantitative:
   sub-topic recall (bipartite chapter↔cluster match by embedding sim / earth-mover);
   article recall (cousin of [[heldout-book-recovery-production-eval]],
   [[book-rediscovery-eval-candidates]]); ordering Kendall-τ on matched chapters. Secondary:
   LLM-judge pairwise (ours vs ToC-as-curriculum) — now the gold arm is a real book.

## Caveats
- **Genre mismatch** (chapter-title+intro vs title+abstract) applies to Track B — cheap
  domain-adaptation fixes before anything learned.
- **Leakage**: hold the book out of the landmark set for Track A. Its *own* gap computation
  self-excludes already.
- **ToC availability** gates the candidate set; report which (book, gap) pairs are in scope.

## Status / working notes (2026-06-01)
Entry point implemented: `artifacts_scripts/curriculum/book_own_gap.py` (querying the whole
book set with `is_intra_cloud=True` gives every book's own gap + the gap-size distribution
in one pass). Validated at **AI-subfield (1702)** scope, radius_fraction=0.75:
- **Popular books have tiny own-gaps.** *Deep Learning on Graphs* (W3160730739, the 81%
  coverer of the T11273 detected gap) → own-gap of only **14 articles**, all near-duplicate
  echoes (other GNN survey books / workshop proceedings). Its nearest non-self book is
  essentially a duplicate → tiny radius. Confirms the gap-vs-established tension: a great
  *coverer* of an article-gap can sit in a *book-dense* region with a near-zero own-gap.
- **Distribution is book-dense**: over 14,591 AI books, median own-gap=0, p90=3, p99=27,
  max=465. Only the top ~1% have a usable own-gap.
- **Top own-gaps split into (a) genuine niche textbooks with coherent gaps** (Federated ML
  on constrained devices 465, Cyberbullying Detection 341, Neuromorphic Photonics 280,
  Semantic Segmentation 141, Multitask Learning 134, Extreme Learning Machine 125) **and
  (b) debris/misclassification outliers** (a Tamil reference grammar → DOI-string garbage; a
  monastery-history book; Indonesian/Thai/Arabic student-thesis clusters).
- **Selection rules learned:** (1) require a *recoverable ToC* — it naturally selects real
  textbooks and drops debris; (2) prefer **large absolute own-gap**, not just high
  percentile (percentile is misleading when the median is 0); (3) **whole-domain scope**
  should shrink the misclassification false-gaps (photonics/crypto books misfiled into AI
  would find true neighbors). Ranked candidates: `artifacts/book_gaps/ai_subfield_ranked.json`.

**Full loop validated end-to-end** (AI scope) on *Automatic Cyberbullying Detection*
(W2898513995, own-gap 341 articles): `book_curriculum.py` (Track A) +
`book_toc_reconstruction.py` (Track B + eval). Results (341 gap → 219 shared with our
15-cluster curriculum; 8 ToC chapters):
- raw GIST-small: ARI=0.074, NMI=0.195, purity=0.30, order-τ=0.05.
- **+ mean-centering** (domain-adapt step 2): ARI=0.112, NMI=0.221, purity=0.36,
  **order-τ=0.34**. Validates the cheap-first sequencing — mean-centering removes the common
  "cyberbullying" component that made generic chapters (Background, Conclusion) absorb
  most articles, rebalancing the partition (Background 120→68, Android-Blocker 0→33) and
  letting our prereq order recover the chapter sequence.
- **Absolute reconstruction is modest** because this is a narrow *research monograph* (the
  authors' own niche methods: web-mining / brute-force / their Android app), not a broad
  field-structure textbook → method-oriented chapters don't match the literature's topic
  clusters. **Next: a broad textbook** (e.g. *Deep Learning on Graphs* would need a
  larger-own-gap niche book; *Quantum Information Theory* T10020 92% is a candidate) where
  ToC chapters reflect field structure. Eval machinery is ready; always apply mean-center.
- Code: `book_own_gap.py`, `book_curriculum.py`, `book_toc_reconstruction.py` (committed
  b867bdb). ToC inputs under `artifacts/book_gaps/*_toc.json` (gitignored).

**Second reconstruction — broad textbook >> monograph (validates the eval).** Mobile
Crowdsensing (W4235013513, Borcea/Talasila/Curtmola; real 10-chapter field-structure ToC),
own-gap 363 articles at CS-field scope. Reconstruction (mean-center): ARI=0.346, NMI=0.457,
purity=0.504 — vs the cyberbullying monograph's 0.112/0.221/0.361. ~3× better on ARI, and the
book's real subtopics map to distinct clusters (Incentives→cl8 37/65, Security/Privacy→cl12
36/66, Systems→cl14 19/28), chapter sizes balanced (no generic-attractor). So the metric is
discriminative and book-type-sensitive; our pipeline recovers ~half a real textbook's
structure. Caveat: ordering-τ is noisy at ~10 chapters (flips with assignment); ARI/NMI/purity
are the robust signals and mean-centering lifts all three. The right target is a focused-field
**textbook** (field-structure ToC), not a narrow research monograph.

**Paired LLM eval — our curriculum vs the book's ToC (the actual goal).** Materialized
Track B as a renderable curriculum run-dir (`book_toc_reconstruction.py --materialize-dir`:
the ToC chapters in order, each populated with its assigned gap-articles + tf-idf keywords,
written as `summary.md` in the curriculum format) so `eval_pairwise_judge` renders it
blind/symmetric vs Track A (labels come from keywords, not the book's real chapter titles;
both show the same neutral blind title). Ran gemma4:31b pairwise (A/B/tie, both orders ×3,
0 parse failures) + a blind opus subagent (both orders) on Mobile Crowdsensing
(ours = Track A base selector, vs book ToC):
- **Both judges: the book's ToC beats our curriculum.** gemma 4-0; opus 3-0 + tie on
  progressive; 3/4 identical winners; **both position-robust (0 flips)**.
- Consistent diagnosis from BOTH (incl. the frontier judge): our gap-seeded curriculum
  **wanders into tangential foundations** — "spends a third of its chapters on tangential
  foundations… general databases, image processing, distributed-systems theory, public-key
  cryptography… a disjointed literature dump" (opus); "wanders significantly into general
  database, cryptography, and algorithm textbooks" (gemma). The PageRank expansion's
  foundations are read as off-topic, costing specificity AND (gemma) progressive; opus
  calls progressive a tie (neither has a clean easy-to-hard arc).
- **This is the headline result the experiment was for**: our *generated* curriculum, head to
  head against a real textbook's structure, loses — primarily on over-broad foundations.
- **Boundary experiment (DONE)**: re-ran Track A with `--boundary` (off-topic-hub demotion)
  and re-judged vs the same book ToC (both phrase-mined, 30 phrases, keywords-only). **The
  judges disagree, and boundary HELPED the frontier judge**: gemma still scores book ToC
  **4-0** (it penalizes any prerequisite as off-topic), but **opus FLIPS** from base (book
  3-0+tie) to boundary (**ours 3-1**, book wins only specificity). Why: boundary *relocates*
  the gap curriculum's foundations from scattered/off-topic (base: database, crypto, image
  processing) to **coherent MCS prerequisites** (networking → sensor networks → MCS), forming
  a real foundations→advanced arc that opus rewards on interestingness/comprehensiveness/
  progressive ("builds explicitly from networking/security/sensor foundations up to MCS
  incentives… a clearer easy-to-hard arc"). Both judges still give **specificity** to the book
  — a gap-seeded curriculum includes prerequisites by construction. **Reading**: gemma is a
  strict "stay exactly on the named topic" judge; opus is a "is this a good *curriculum*"
  judge that credits a coherent prerequisite ramp — arguably the more appropriate frame for a
  curriculum. So boundary's value is making the prerequisites *coherent*, not removing them.
  Boundary results: gemma `mcs_boundary_vs_toc.json`, opus `.../opus_probe/mcs_boundary_opus.json`.
  Results: gemma `artifacts/curriculum_eval_pairwise/mcs_ours_vs_toc.json`, opus
  `.../opus_probe/mcs_opus.json`.
- **Render refinements (robustness checks).** (a) **Keywords-only** (drop the noisy sample
  articles): verdict UNCHANGED for both judges (gemma 4-0, opus 3-0+tie) — the keyword
  labels already carry the foundation-wandering signal, so we adopt keywords-only
  (`--n-titles 0`). (b) **Both phrase-mined + 30 phrases/chapter** (fair: Track B was
  routed through the curriculum's `extract_topic_keywords` phrase miner instead of tf-idf;
  Track A re-extracted at top_k=30; `render_outline` shows 30; `NUM_KEYWORDS_IN_SUMMARY`
  10→30): final CLEAN verdict gemma **4-0**, opus **3-0 + comprehensiveness tie** — the book
  ToC still wins. A transient "ours wins comprehensiveness" appeared only when Track B was
  materialized with the RAW (non-mean-center) assignment, which left its Incentive-Mechanisms
  chapter EMPTY; with mean-center (all 10 chapters populated) it reverts to book-ToC. opus on
  the clean run: "ALPHA's extra chapters are mostly off-topic padding rather than added MCS
  coverage" — our 15-vs-10 breadth is padding, not comprehensiveness. **Conclusion is robust
  to judge, article inclusion, keyword method, and keyword count.** Always materialize Track B
  with `--mean-center`. Final: gemma `mcs_final.json`, opus `.../opus_probe/mcs_opus_final.json`.

**Lean book-first path (the intended workflow).** `book_own_gap.py --brute-force` computes ONLY
the chosen book's gap exactly — nearest non-self book + article ball via dot-products, no
all-books query, no HNSW build/discard. Exact (the HNSW path under-finds: 349 vs 341 on
cyberbullying, 363 vs 345 on MCS) and fast (~80s compute; ~2-3 min incl. cloud load at CS
scope, vs ~28 min for the all-books `--rank-top`). Use `--rank-top` only for candidate
DISCOVERY; for a known OA book, `--brute-force`. Whole-domain single-book gaps are now feasible
in minutes (Lever A `SubCloud.embeddings()` + brute-force; the HNSW path's 35M index is never
persisted anyway — `build_index` is in-memory only).

**Scope findings (AI subfield vs CS field vs whole domain).** Ran the own-gap ranking at
all three (whole-domain needed the `perf(cloud)` row-selective `SubCloud.embeddings()` fix,
commit bab2cc8 — before it, domain scope OOM'd materializing the full 58.5 GiB parent array;
after, it peaked ~122 GB and completed in 2h22m). Key result: **broadening scope makes the
debris problem WORSE, not better** (contra the earlier hypothesis). Whole-domain top own-gaps
are ~37/40 non-English misclassified textbooks (Russian/Ukrainian "учебное пособие", Korean
policy reports, Chinese teaching books) that are extreme embedding outliers (gaps 1000–3000).
Meanwhile a genuine focused-field book's own-gap is **scope-robust** — Federated ML
W6931779801 = 465 (AI) / 485 (CS) / 496 (domain), always its true federated-learning cluster.
**Consequences:** (1) compute a genuine target's gap at the CHEAPEST scope (AI subfield) — it's
~the same; (2) candidate selection should stay at subfield/field scope (the scope itself filters
the off-domain debris) — whole-domain ranking is debris-dominated and a poor candidate source;
(3) the real lever for the debris is a data-quality filter (non-English / ToC-recoverability /
binomial-p), not scope. Rankings: `artifacts/book_gaps/{ai_subfield,csfield,domain}_ranked.json`.

## Cross-refs
- [[curriculum-generalization-pedagogical-subspace]] — alignment machinery for Track B.
- [[curriculum-diffusion-and-quality-metrics]] — LectureBank source + intrinsic metrics.
- [[curriculum-ranking-input-levers]] — the structuring the reconstruction score isolates.
- Code: `bookgap/gaps.py`, `bookgap/llm/gap_algorithms/kball_algorithm.py`,
  `bookgap/gap_statistics.py`, `artifacts_scripts/curriculum/seed_from_gap.py`;
  candidate scan saved at `/tmp/toc_candidates.json`.
