# Curriculum ToC-matching via dendrogram cuts

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
within ~0.01–0.02 on every book, so clustering choice is second-order; (b) the ToC is **never a
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

## What's settled / what's next

- **Matching:** use UOT, not Hungarian — it dissolves the spurious cut-method ranking and reads
  generality off the transport plan. The cut method (effN_sil vs flat KMeans) is second-order.
- **Generality / intro chapters:** both UOT entropy and general-node recovery identify overview
  chapters a flat partition can't represent — the diagnostic half of the cycle idea, without the
  cycle machinery. The *generative* half (emit a ToC with a real intro node + Euler-tour order)
  is still open.
- **Ordering:** human ToCs sequence by **local good edges, never a global path** — but **only
  pedagogical single-author textbooks do** (Jurafsky 98–100th pctile). Edited research volumes
  don't (≤40th even at high corpus-fit, post-causal-test). Good-edges looks like a **genre signal**
  (sequenced monograph vs edited collection), *not* a corpus-fit effect — the causal test killed
  the fit interpretation.
- **Open (confirm genre):** the genre hypothesis is still confounded with chapter-count power
  (textbook n=25 vs volumes n=8–13). Test a **single-author pedagogical monograph in a niche field
  with many chapters** (e.g. Koehn's SMT, Bishop, Goodfellow) — if it shows good edges at its own
  fit, genre is confirmed over both fit and n.
- **Still open (generative):** emit an algorithmic ToC with a real intro node + Euler-tour order
  (the generative half of the cycle idea); UOT/recovery only *diagnose* generality so far.
