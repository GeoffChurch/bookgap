# Curriculum ToC-matching via dendrogram cuts

> **Note — tooling removed.** The `artifacts_scripts/curriculum/` modules referenced in this doc were deleted in `0f2d0ad`; findings stand, commands do not. See `docs/curriculum-pipeline-removed.md`.

*Does a field's citation/topic structure predict a human textbook's table of contents — its
chapter **selection**, its **granularity**, and its **ordering**? And which clustering/cut method
best recovers it?* This extends the gap-based ToC-reconstruction eval
([`curriculum-toc-reconstruction-eval.md`](curriculum-toc-reconstruction-eval.md)) to
**comprehensive** textbooks, which have empty own-gaps and are structurally excluded from the
gap pipeline. Here the corpus is the **field** (subfield-level OpenAlex slice), not the book's
own-gap ball.

Harness: `artifacts_scripts/curriculum/toc_dendrogram_cut.py` (+ `fetch_subfield_corpus.py`).
Run outputs under `artifacts/toc_dendrogram/`.

## Pipeline

1. **Corpus** — top-K most-cited works of the book's field. Jurafsky uses the cached NLP topic
   (`nlp_t10181`); the niche books use `fetch_subfield_corpus.py`, which pulls the book's
   **subfield** (OpenAlex book `primary_topic` tags are noisy; subfield is reliable). Text =
   reconstructed abstracts; node mass = `cited_by_count`.
2. **Filter** — register/language cleanup at two levels: drop non-English abstracts (English
   function-word fraction) and proceedings front-matter (works); drop venue acronyms, years,
   ordinals, DOIs (tokens). Without it ~6–8 of 25 topics are venue/author/DOI debris.
3. **Embed** — TF-IDF → LSA-64 (the clustering space).
4. **Hierarchy + cut** — Ward linkage, cut to N topics by a pluggable max-heap key:
   `height` (= the standard flat dendrogram cut), `mass`, `effN` (effective #works), and
   **`effN_sil`** = effN × child-silhouette (the adaptive "split heterogeneous-but-separable,
   leave cohesive coarse" cut). **`adaptive`** lets the data pick N\* (size-capped: split the
   biggest over-cap cluster until none dominates — a balance-floor degenerates to N=2 because
   citation mass is power-law). `kmeans` is the flat baseline.
5. **Reference** — materialize the human ToC as a partition of the same works: ST-embed each
   chapter's title+description (MiniLM, a space **independent** of the LSA clustering geometry —
   this de-confounds KMeans, which otherwise shares the nearest-centroid objective) and assign
   each work to its nearest chapter.
6. **Score** — `ruzicka` (mass-weighted Jaccard) + **Hungarian** 1-1 alignment, **UOT**
   (unbalanced Sinkhorn) transported similarity, best-match **coverage**, per-chapter
   **generality** (= effective #topics feeding a chapter under UOT; high ⇒ overview chapter),
   **path-likeness** (Spearman of chapter# vs dendrogram leaf-order), **good-edges** (do the
   human's consecutive chapters land on high-similarity topic pairs vs random orderings), and
   **general-node recovery** (add broad internal nodes as candidates so overview chapters can
   match a general node).

## Findings — Jurafsky (Speech and Language Processing, 25 ch, NLP field)

**Selection.** The cut recovers the field's real research areas — MT, Embeddings, IR, IE,
Semantic parsing, Summarization, Dependency parsing, Morphology, Constituency, Discourse, WSD,
Phonology — but only **16 of 25** chapters get a *distinct* nearest topic. The 9 misses are the
foundational/recent/general chapters (Intro, Post-training, Test-time compute) that aren't
citation clusters. Adaptive-N picks **N\*=28 ≈ 25** — the field's natural granularity ≈ the
textbook's chapter count.

**Matching (Hungarian → UOT).** Under rigid 1-1 Hungarian, flat KMeans appeared to beat the
adaptive cut (0.261 vs 0.204 mean matched ruzicka). Half that gap was a **confound** — the
LSA-centroid materialization shares KMeans's objective; under the neutral ST materialization the
gap halved (0.210 vs 0.179). And under **UOT** (soft, many-to-many) the gap **vanishes**:
effN_sil 0.566 ≈ adaptive 0.560 ≈ kmeans 0.573. **The clustering method barely matters; the
rigid alignment was the artifact.**

**Generality (the intro problem).** A flat partition cannot represent an overview chapter (it
overlaps every specific chapter). UOT's per-chapter transport entropy reads generality off for
free: **general** = Words & Tokens, Neural Networks, LLMs, Logistic Regression (eff ≈ 13–15
topics feed in); **specific** = Embeddings, RNNs, MT, Transformers, Dependency Parsing (eff ≈
2–5). **General-node recovery** independently agrees: adding broad nodes routes 4 chapters
(Words & Tokens, N-grams, LLMs, Post-training) to a general node and lifts distinct coverage
16→17/25. Two methods flag the same chapters as general.

**Ordering.** The ToC is **not** a global path through the dendrogram (chapter# vs leaf-order
Spearman null, p≈0.45). But its **local transitions follow good edges**: consecutive chapters
land on topic pairs more similar than **98–100%** of random chapter orderings. Humans sequence by
local topical adjacency, not a global sweep.

## Generalization (5 books, 5 fields)

Corpora pulled at subfield level (4000 top-cited works each), **except Jurafsky** uses the cached
NLP *topic* `nlp_t10181` (tighter than a subfield — see the resolution caveat below). Books sorted
by corpus-fit (distinct-coverage):

| book (field) | ch | N\* | UOT effN_sil/adapt/kmeans | distinct (fit) | good-edge pctile |
|---|---|---|---|---|---|
| Jurafsky SLP (NLP topic) | 25 | 28 | 0.566 / 0.560 / 0.573 | 16/25 = .64 | **98–100%** |
| Cyberbullying (subfield) | 8 | 32 | 0.232 / 0.207 / 0.232 | 3–4/8 ≈ .44 | 45–79% |
| Color Image (CV subfield) | 24 | 30 | 0.349 / 0.351 / 0.358 | 8–10/24 ≈ .38 | 62–89% |
| Mobile Crowdsensing (subfield) | 10 | 31 | 0.291 / 0.280 / 0.287 | 1–3/10 ≈ .20 | 0–14% |
| Neuromorphic Photonics (subfield) | 13 | 32 | 0.206 / 0.192 / 0.198 | 2–3/13 ≈ .19 | 0–31% |

**Replicates across all five:** (a) **UOT ties the cut methods** — effN_sil ≈ adaptive ≈ kmeans
within ~0.01–0.02 on every book, so clustering choice is second-order *under UOT* (the *linkage* does matter on the real recon metric — see `docs/curriculum-dendrogram-quality.md`); (b) the ToC is **never a
global path** — chapter# vs leaf-order Spearman is null on all five (largest |ρ|=0.51 at n=8,
p=0.19).

**N\* reflects field granularity, not chapter count.** Adaptive N\* ≈ the human count only for
**field-spanning** textbooks (Jurafsky 28≈25, Color Image 30≈24). The niche books gave N\*≈31–32
vs 8–13 chapters: a narrow book covers a small corner of a broad field, so the field has far more
topics than the book has chapters. N\* measures the corpus, not the book.

**Good-edges *appeared* to track corpus-fit** — across the subfield runs it looked monotone in
distinct-coverage (Jurafsky .64 → 98–100th pctile; Crowdsensing/Neuromorphic .19–.20 → ≤31st).
That motivated a causal test.

**Causal test (tight per-book corpora) — FALSIFIES the corpus-fit law.** Re-running the three niche
books on their own gap-ball neighborhoods (`fetch_ids_corpus.py`, ~270–430 works each) raised fit
sharply — UOT .20→.68 (Neuromorphic), .29→.68 (Crowdsensing); distinct 2–3→5–9, 1–3→5–7 — but
**good-edges did NOT rise**: Neuromorphic 0–31→22–33rd, Crowdsensing 0–14→9–39th, Color Image even
*fell* 62–89→24–68th (still at/below random). So improving corpus-fit does **not** buy ordering
signal; the subfield "good-edges ∝ fit" pattern was a **confound**.

**The leading explanation is genre, not fit.** Jurafsky is a single-author *pedagogical* textbook —
deliberately sequenced, so consecutive chapters are topically adjacent (good edges). The four niche
titles are *edited research volumes* (Color Image Processing, Neuromorphic Photonics, Cyberbullying
Detection, Mobile Crowdsensing): author-independent chapters, topic-grouped but not pedagogically
ordered → no good edges even at high fit. (Co-caveat: 8–13 chapters also underpowers the
permutation test — too few transitions.) The corpus-fit gap was real for *fit* (topic vs subfield
resolution) but is **not** what drives good-edges.

## Genre confirmation (Koehn) + generative ToC

**Koehn's SMT (single-author monograph, 11 ch) — moderate genre support.** Run on the NLP topic
corpus (same as Jurafsky; Koehn's `primary_topic` is also T10181). At the **same chapter count as
the edited volumes** (8–13), the monograph shows ordering structure the volumes lacked: adaptive
cut good-edge **83rd pctile** and a *significant global path* (Spearman +0.64, p=0.04), vs the
volumes' ≤40th and nulls. So genre beats n-power — *but* it's cut-sensitive (effN_sil@11 only 29th)
and weaker than Jurafsky, because the broad NLP corpus is a medium fit for an SMT-specific book.
Not a slam dunk; directionally confirms genre.

**Generative ToC (`--generate`): good-edge order recovers the human sequence.** A greedy
good-edge walk over the topics (the leaf-order/Euler-tour is null, so order by local adjacency
instead) + a general intro node:
- **Generated order aligns with Jurafsky's human order: Kendall-τ +0.39, p=0.008** — where the
  naive leaf-order was null (−0.16). Humans order by *local good edges*, and a greedy good-edge
  generator reproduces ~39% of the human sequence. (Koehn τ=−0.26, p=0.31 — null, medium fit + n=11.)
- Generated orders are **more locally coherent than the human's** (good-edge weight 0.805 vs 0.776
  Jurafsky; 0.773 vs 0.730 Koehn) — the human trades some local coherence for pedagogical
  dependency/narrative.
- **The intro-node heuristic FAILED** — "broadest internal node" landed on N-gram LMs (Jurafsky) /
  Language Models (Koehn), not the Introduction. Size ≠ intro.

**Follow-ups (a) centrality intro + (b) high-fit corpus — corpus-fit is the master variable.**
Replacing size with a **centrality** selector (the broad node most similar on average to all topic
centroids) and re-running Koehn on an **MT-tight** corpus (the NLP corpus filtered to ≥2 MT terms,
3,963 works — a tight fit to its SMT scope):
- **High fit fixes both at once.** On the MT-tight corpus the centrality intro lands on **ch1
  Introduction (match)**, and order-recovery **flips from null to positive: Kendall-τ −0.26 →
  +0.32** (p=0.195; n=11 underpowers significance, but the sign now matches Jurafsky). Higher fit
  strengthens order-recovery, as predicted.
- **Centrality alone, on the broad corpus, is not enough** — it still missed ch1 on Jurafsky (→ch3)
  and Koehn/NLP (→ch7): the full NLP field's residual debris ("sciendo, professional
  organizations") pollutes the central node. The selector is correct in principle (works on the
  clean tight corpus) but **corpus-fit-gated** — like good-edges and order-recovery before it.
  Corpus-fit/cleanliness governs the whole study: it sets whether good-edges is measurable, whether
  the intro resolves, and whether the generated order recovers the human's.

## What's settled / what's next

- **Matching:** use UOT, not Hungarian — it dissolves the spurious cut-method ranking and reads
  generality off the transport plan. The cut method (effN_sil vs flat KMeans) is second-order *under UOT*. **Refined (2026-06-15):** that verdict is **metric-specific** — soft OT flat-matching is blind to tree shape; on the **real close-loop recon metric** the *linkage* (tree construction) is NOT second-order: **average-linkage beats the bisecting default by +0.137 ARI (positive 4/4 books)**, tops Ward/bisecting on cophenetic fidelity, and is now the curriculum default. See `docs/curriculum-dendrogram-quality.md`.
- **Generality / intro chapters:** both UOT entropy and general-node recovery identify overview
  chapters a flat partition can't represent — the diagnostic half of the cycle idea, without the
  cycle machinery. The *generative* half (emit a ToC with a real intro node + Euler-tour order)
  is still open.
- **Ordering:** human ToCs sequence by **local good edges, never a global path** — but **only
  pedagogical single-author textbooks do** (Jurafsky 98–100th pctile). Edited research volumes
  don't (≤40th even at high corpus-fit, post-causal-test). Good-edges looks like a **genre signal**
  (sequenced monograph vs edited collection), *not* a corpus-fit effect — the causal test killed
  the fit interpretation.
- **Genre (Koehn, done):** moderately confirmed over n-power — the n=11 monograph shows ordering
  structure (adaptive 83rd pctile, path p=0.04) the same-n edited volumes lacked. Cut-sensitive;
  a higher-fit monograph corpus (or Bishop/Goodfellow) would sharpen it.
- **Generative (done, the keeper):** a greedy **good-edge** order — not the leaf-order — recovers
  Jurafsky's human chapter sequence (Kendall-τ +0.39, p=0.008). The generative half of the cycle
  idea works for the ordering. Two fixes open: a real **intro-node selector** (centrality, not
  size — the size heuristic missed ch1 on both books) and re-testing generation at high corpus-fit.
