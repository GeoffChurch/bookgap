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

**Good-edges tracks corpus-fit — the headline.** The ordering signal is **monotone in
distinct-coverage**: well-resolved books show it strongly (Jurafsky .64 → 98–100th pctile),
poorly-resolved ones drop to **random** (Crowdsensing/Neuromorphic .19–.20 → ≤31st pctile). Below
~.3 distinct-coverage the human's "good local transitions" are indistinguishable from a random
chapter order — not because the human is disordered, but because the corpus lacks the topics to
detect it. UOT-similarity falls the same way (.57 → .20). So "human ToCs follow good edges" holds
**when the corpus is the right field at the right resolution**, not unconditionally.

**Resolution caveat (honest).** Part of Jurafsky's superior fit is that it used a *topic*-level
corpus while the niche books used broader *subfield* corpora (their OpenAlex `primary_topic` tags
were too noisy to trust — Color Image's resolved to "blood-disease imaging"). So fit is confounded
with corpus-resolution. The clean next test is a **tighter per-book corpus** (the book's
referenced-works neighborhood) — if good-edges strengthens with fit there, the corpus-fit law is
causal.

## What's settled / what's next

- **Matching:** use UOT, not Hungarian — it dissolves the spurious cut-method ranking and reads
  generality off the transport plan. The cut method (effN_sil vs flat KMeans) is second-order.
- **Generality / intro chapters:** both UOT entropy and general-node recovery identify overview
  chapters a flat partition can't represent — the diagnostic half of the cycle idea, without the
  cycle machinery. The *generative* half (emit a ToC with a real intro node + Euler-tour order)
  is still open.
- **Ordering:** human ToCs sequence by **local good edges, never a global path** — but the signal
  is only measurable when the corpus resolves the book (good-edges ∝ distinct-coverage across all
  five books; ≤~.3 fit ⇒ random). This is as much a measurement law as a finding about humans.
- **Open (the causal test):** corpus-fit is the lever, and it's currently confounded with
  topic-vs-subfield resolution. Pull each book's **referenced-works neighborhood** (a tight,
  book-specific field) and re-measure: if good-edges climbs with fit on the *same* books, the
  corpus-fit law is causal, and the niche-book "nulls" are corpus artifacts, not disordered ToCs.
- **Still open (generative):** emit an algorithmic ToC with a real intro node + Euler-tour order
  (the generative half of the cycle idea); UOT/recovery only *diagnose* generality so far.
