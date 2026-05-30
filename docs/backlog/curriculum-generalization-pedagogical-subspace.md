# Curriculum engine → standalone, input-agnostic project (+ Wikipedia graph, + CCA "pedagogical subspace")

Lift the curriculum machinery out of bookgap into a **standalone project** whose
input is pluggable, and explore non-citation graphs (Wikipedia) plus a learned
"pedagogical" subspace as ways to get better curricula.

## Idea

The curriculum engine is already source-agnostic in principle: it needs only a
**(sparse graph, per-node embeddings, seed nodes)** triple, then does seeded
ranking → top-k clustering → linearization → keywords → summary. The
bookgap-specific parts (`Cloud`, OpenAlex scopes, dedup) are just *one adapter*.
A clean `GraphSource` interface (→ adjacency, node embeddings, id↔index,
node metadata) would make new corpora drop-in.

First-class backends to support:
- **OpenAlex citation graph** (current — see `docs/backlog/subjective-curriculum-eval.md`).
- **Wikipedia link graph** (+ article embeddings).
- Later: arXiv, textbook/syllabus structure graphs, course catalogs.

## Why Wikipedia might give *better* curricula

Wikipedia is **explicitly pedagogical**: its link structure and article
granularity encode "what you need to know to understand X" much more directly
than a citation graph, which encodes "what this paper built on" — biased toward
recent/citable work, with acknowledgment/boilerplate noise and broad
cross-field citing.

Concrete evidence from the OpenAlex-citation run (the 5 curricula in
`subjective-curriculum-eval.md`): the curricula were good, but the **foundation**
clusters were diffuse and showed citation-graph artifacts —
- an **off-topic signal-processing ("PCM distortion") cluster** intruding into the
  Intelligent-Tutoring curriculum (forward-PageRank pulled in a broad generic-AI
  neighborhood),
- **boilerplate keywords** (`thank, thesis, ideas`) from thesis front-matter,
- generic foundation-topic keywords (`theory, first, system`).

A pedagogical graph would likely avoid these. Hypothesis: **Wikipedia → better
curricula for established topics.**

## The catch, and the best-of-both fix

Wikipedia has far fewer articles than OpenAlex has works, and lags badly on
**niche / cutting-edge research** (exactly the gaps this project targets).
OpenAlex covers the frontier and the long tail; Wikipedia covers the canon.

**Proposed best-of-both — a learned "pedagogical subspace":** align OpenAlex
embeddings to Wikipedia's pedagogical structure via **CCA** (or Procrustes /
paired-anchor alignment) on items present in *both* corpora (anchor pairs:
OpenAlex work ↔ its Wikipedia article, or topic-matched centroids). Then run
curricula on **OpenAlex** (keeping its coverage + frontier) but in the
**pedagogically-aligned subspace**, so ranking/clustering/ordering inherit
Wikipedia's "prerequisite" geometry. Net goal: Wikipedia's pedagogical quality
*and* OpenAlex's coverage of niche/new work.

bookgap already has CCA-adjacent corpus-alignment machinery — the **CCLIP / GEP
family** in `bookgap/denoisers.py` + `bookgap/cloud_denoiser.py` (it aligns two
corpora into a shared subspace, currently books vs articles). That's the natural
starting point / inspiration for the pedagogical alignment.

## Sub-questions / variants

- **Graph vs embeddings:** use Wikipedia's *link graph* for ranking, vs only its
  *embeddings* for the subspace, vs both.
- **Anchor set for CCA:** DOI/title matches, OpenAlex↔Wikipedia entity links, or
  topic-cluster centroids; how many anchors are needed.
- **Does the pedagogical subspace also fix keyword extraction** (sharper
  foundation-topic keywords, less boilerplate)?
- **Evaluation:** re-run the 5 curricula from `subjective-curriculum-eval.md` in
  the pedagogical subspace — does T11902 lose its PCM/signal-processing intrusion?
  Does ascending Sim→Target tighten?

## Pointers

- Current engine: `artifacts_scripts/curriculum/` (`build_curriculum.py`,
  `pagerank.py`, `config.py`, `seed_from_gap.py`).
- Corpus-alignment machinery to repurpose: `bookgap/denoisers.py` (CCLIP/GEP),
  `bookgap/cloud_denoiser.py`.
- Curricula this would generalize/improve: `docs/backlog/subjective-curriculum-eval.md`.

## Cached Wikipedia data in `~/src/bookgen` (scouted 2026-05-29)

Reusable assets exist, but **embeddings are the gating limit**:

- ✅ **Full enwiki link graph** (dump `enwiki-20260201`): `data/wikidumps/.../{page,
  pagelinks,categorylinks,linktarget}.sql.gz` + parsed pickle caches. Standalone loaders:
  `src/eval/topics/dump_parser.py` (`load_page_table`, `load_pagelinks`, …) and
  `src/eval/topics/wiki_graph.py:build_link_subgraph(edges, article_ids) → (scipy CSR,
  id_list[page_id])`; title↔id via `src/eval/wiki_corpus.py:WikiCorpus`. → usable as a
  `GraphSource` (graph + id-map ready; also a category DAG for scoping).
- ⚠️ **Embeddings cover only 879 articles** — `data/rag_index/` (LlamaIndex), model
  **BAAI/bge-small-en-v1.5, 384-D, chunk-level** (7,125 chunks; page_id recoverable from
  node metadata). NOT enwiki-wide, and a *different* model than bookgap's GIST/Gemma.
- ✅ Cleaned text for those 879: `data/corpus/{page_id}.txt` + `manifest.json`.

**Implications:**
- *Direct Wikipedia curricula:* graph is enwiki-scale but embeddings exist for only 879
  articles → ready only at that small scope; enwiki-scale needs embedding the bulk (a real
  GPU job, ~hours). Cleanest quick probe: embed a targeted Wikipedia slice (e.g. CS /
  physical-sciences via the category graph) **with GIST-small** so it shares bookgap's
  space (no CCA needed for direct use).
- *CCA pedagogical subspace:* hard anchors are capped by 879 embedded articles ∩
  OpenAlex-matchable (DOI / title / Wikidata) → realistically tens-to-low-hundreds — small
  but maybe enough for a prototype. Wikidata/DOI linking is the setup cost.

## Status / working notes

**2026-05-30 — first direct Wikipedia curriculum built + run (cirrus-native).**
Superseded the `~/src/bookgen` path above: rather than the `enwiki-20260201`
pagelinks SQL dump (which forced a page_id intersection across two dumps), we
build the link graph from the **same** `enwiki-20251229` CirrusSearch *content*
dump that produced the GIST embeddings — `outgoing_link` carries the graph
natively, so it's internally coherent (one date, no cross-dump mismatch, no
bookgen). New code: `bookgap/wikipedia/link_graph.py` (+ test),
`artifacts_scripts/wiki/wiki_curriculum_demo.py`. Caches `title_maps.parquet`
+ `curriculum_subgraph_n40000.npz` so re-runs skip the 43 GB dump
(`--reuse-subgraph`).

**Result (full write-up: `docs/wikipedia-curriculum.md`): worse than the
OpenAlex-citation curriculum as-is — bibliographic-infrastructure hub
contamination.** Forward PageRank on the raw link graph put **DOI / ISBN /
PubMed / ArXiv / ISO / Wikidata** at the top (linked from ~every article via
citation templates), outranking the real NLP foundations; the six seeds each
collapsed to a singleton cluster below them; frontier (Transformers/BERT) barely
appeared. So the hypothesis "Wikipedia → better curricula" is **not confirmed for
the raw link graph** — but topics 1–2 (CS/AI/Algorithm/Turing; Formal
grammar/Compiler) are sensible foundations, so there's real signal under the
noise.

**Next (the actual open idea now):** hub suppression before this is worth
pursuing as the pedagogical graph. Three salvage paths, cheapest first:
1. **Background-normalized PageRank** — reuse the existing `p^a·log(p/g)`
   background normalization (built for the OpenAlex curriculum): DOI/ISBN/PubMed
   have huge background (un-seeded) mass → demoted automatically. Most principled,
   reuses machinery.
2. **Infrastructure category filter** — the cirrus dump carries each page's
   `category`; drop identifier/database/standards-org pages in a one-pass filter.
3. **In-degree hub suppression** — downweight/drop edges into extreme-in-degree
   targets.
The CCA "pedagogical subspace" idea (above) is untouched and still open; this
note only covers the *direct Wikipedia link-graph curriculum* probe.

### 2026-05-30 (later) — wiki↔OA shared-space geometry measured; alignment design space

Before building the pedagogical subspace we measured whether wiki and OA actually
*share geometry* (same GIST encoder, so naively "yes"). Full results:
`docs/wikipedia-openalex-shared-space.md` (§"Is the shift a pure offset?" + "Topic-matched re-test").

**Finding: NOT a pure offset — a genuine *nonlinear* register residual, even within one
topic.** After per-corpus mean-centering the difference is linearly inseparable (AUC ≈ 0.53,
no LDA direction recovers it), but covariance differs materially (PC-subspace principal
angle 41° at k=50, topic-matched) and local kNN same-corpus purity stays 0.92 (37σ MMD
above the within-corpus null). Topic-matching the wiki sample (NLP-wiki vs NLP-OA, not NLP
vs all-of-Wikipedia) shrinks every metric but only ~13% of the clumping excess was
topic-distribution — the rest is corpus/register.

**Implications for the alignment (this revises the plan):**
- **LPP-on-the-link-graph** (concrete, anchor-free): use the Wikipedia *link adjacency* as
  the LPP affinity `W` (not kNN-of-embeddings) — solve `X L Xᵀ a = λ X D Xᵀ a` (L = D−W);
  bottom non-trivial eigenvectors are directions where *linked* articles stay close →
  "link-smooth" pedagogical axes. Cheap (384×384 eigenproblem; only the sparse `X·L` matmul
  costs). Needs **no OA↔wiki anchor pairs** (unlike CCA). **Hub suppression essential** — the
  Laplacian inherits PageRank's hub pathology (DOI/ISBN dominate degree), so weight edges by
  the background-mass / inverse-target-in-degree `g` that already fixed the curriculum
  PageRank (one hub-suppression scheme, two uses).
- **But naive LPP-transfer (learn P on wiki, apply to OA) will degrade** — the measured
  covariance mismatch means wiki-geometry ≠ OA-geometry, so a wiki-learned linear projection
  is partly mis-specified on OA. The shift battery *bounds* the transfer; this walks back the
  earlier "LPP beats CCA because it's anchor-free" lean (the anchor-free transfer is well-posed
  only under shared geometry, which we don't have).
- So the design splits: (a) for wiki's *own* pedagogical structure, LPP-on-graph is great;
  (b) for **best-of-both transfer to OA**, align the second moment first — **CORAL** (whiten/
  recolor to match covariance; cheap, no anchors) or **CCA** (aligns shared directions; needs
  anchors) — *then* apply the link-graph LPP. A **joint** formulation (learn pedagogical axes
  over both corpora at once, links supplying wiki affinity) is the principled version.
- **Per-slice mean-matching caveat:** the pipeline removes the *global* per-corpus offset, but
  a non-representative slice (NLP-only) keeps a residual mean (norm 0.39) — any wiki→OA map
  must mean-match per slice (or jointly) before covariance alignment can help.
- **mycooc/coocmap angle — first verdict RETRACTED (2026-05-30); re-evaluating.** An initial
  assessment mischaracterized coocmap as VecMap/Procrustes (an orthogonal-map self-learning loop)
  and dismissed it as "can't fix a nonlinear residual." **That was wrong.** coocmap/softcoocmap
  does NOT learn an orthogonal map: it uses cross-domain NN to define **anchors**, re-represents
  every point in every corpus by its similarity profile to those anchors (an "anchor space" of
  dim = #anchors, summable across ≥2 corpora), recomputes NNs, and iterates — a **nonlinear,
  non-parametric** re-representation. The "orthogonal map can't help" objection therefore does
  **not** apply; coocmap's anchor-space view may handle exactly the nonlinear/covariance residual
  the *linear* methods (CORAL/CCA/LPP-transfer/Procrustes) are bounded by — potentially the **most**
  promising approach, not the least.
- **Feeding it is feasible** (the prior "no dense input" objection is moot): mycooc consumes a
  V×V association matrix; our analog is the **gram matrix (pairwise cosine), sparsified to kNN** —
  build per-corpus kNN-gram (reuse `build_similarity_adjacency_from_cloud`), feed wiki + OA grams.
  Shared encoder ⇒ start from the identity mutual-NN (skip the sorted-signature init). A
  corrected-understanding subagent is re-assessing + may prototype. CSLS (the one already-harvested
  primitive) stays ported (`_gep_csls`; `keyword-extraction/soft-hubness-correction-csls.md`).
- **Back-flow to mycooc:** if mycooc ever aligns corpora that already share an encoder, it should
  skip the sorted-signature init and start from the identity NN dictionary — a clean dual of this
  finding, worth a note on the mycooc side *if* they add a dense-embedding input path (none today).

**Alignment approaches to track (don't lose these).** Shared preprocessing: per-slice mean-match.
1. **LPP on the wiki link graph** — wiki's *own* pedagogical subspace (anchor-free).
   **VALIDATED 2026-05-30** (`docs/wikipedia-pedagogical-lpp.md`, `e8f978e`): hub-suppressed LPP
   with the *link-graph* affinity at k≈64 raises link-modularity +23% (Q 0.240→0.295) and
   link-flow capture +24% over raw GIST (same selection + clusterer), demotes DOI/ISSN/ArXiv
   infra out of the top AI/ML topic, and surfaces CS/statistics/math-logic foundations. Crucially
   attributable to the **human-curated links specifically**: link-LPP and kNN-of-embeddings-LPP
   subspaces are near-orthogonal (mean canonical corr **0.016**) and kNN-LPP does *not* improve
   link-clustering — so the gain is the links, not generic graph regularization. Captures
   *community*, not prerequisite *direction* (ordering stays graph-based: frontier-adjacency /
   voltage). k must be re-swept per scope (k=32 underfits, k=128 regresses to baseline).
2. **coocmap anchor-space** — **REFUTED 2026-05-30**, now correctly (verified intra-cloud
   relational loop + correspondence metric; `docs/dead_ends/shared-space-alignment/coocmap-anchor-space-wiki-oa.md`).
   It doesn't recover wiki↔OA correspondences and is beaten badly by the raw shared-encoder cosine
   NN (top-20 precision ~0.90 vs ~0.15–0.40; MNN 0.373→0.13, *collapses* instead of climbing). The
   shared encoder already solves the matching (no cold-start), and wiki/OA intra-cloud relational
   structure isn't analogous (register residual).
3. **CSLS-mutual-NN pseudo-anchors → CORAL** — **TESTED, don't bake in 2026-05-30** (`a973699`;
   bake-off in `docs/wikipedia-openalex-shared-space.md`). Halves the kNN-purity excess
   (0.917→0.773) at ~0.95 NN precision, but **relocates** the residual rather than reducing it —
   inflates global MMD² 18σ→82σ (recoloring the whole OA cloud from a 556-anchor covariance).
   Lateral, not a win.
4. **CSLS-mutual-NN pseudo-anchors → CCA** — **TESTED, refuted 2026-05-30** (same). The battery
   "improves" only via the trivial by-construction angle→0° squeeze; NN precision **craters to
   ~0.20** (string/homonym overfitting, e.g. Register-music→register corpora).

**Cross-corpus alignment — CLOSED (2026-05-30).** The usable cross-corpus signal is the **raw
mean-matched shared-encoder cosine dictionary** (~0.85 top-20 precision; **CSLS** refines it to
~0.95 and ~2× mutual-NN) — which resolves the coocmap "~0.90" ambiguity: it was *raw*, and CSLS is
a clean +0.10 hubness refinement, not the source. No learned alignment earns its keep — coocmap is
beaten by raw, CORAL relocates the residual, CCA craters precision. The nonlinear register residual
is genuine and linear second-moment alignment can't resolve it without sacrificing the
correspondence. **Recommendation: use raw (or CSLS-refined) mean-matched cosine for cross-corpus
correspondence; do NOT bake CORAL/CCA/coocmap into the cross-corpus path.** The pedagogical win is
wiki-native (LPP, item 1).

### The realistic best-of-both path: dictionary-bridged joint graph (2026-05-30)

Since cross-corpus *space alignment* is bounded (above) but the cross-corpus *correspondence
dictionary* is strong (raw ~0.85 / CSLS ~0.95), realize best-of-both **graph-side, not
embedding-side**: build a JOINT graph over `wiki articles ∪ OA works` with three edge types —
**wiki links** (pedagogical structure), **OA citations** (coverage + frontier), and **cross-corpus
bridge edges from the cosine dictionary** — then run the validated wiki-native pipeline
(hub-suppressed PageRank + link-LPP + curriculum) on the union. The dictionary is bridge *edges*,
not a space map, so this sidesteps the alignment problem entirely.

**Precision is NOT the binding constraint.** The 0.95 covers the *overlap* (canonical concepts in
both corpora) — but the dream's payload is OA-only niche/frontier work with *no* wiki counterpart,
which has no dictionary entry, and mutual-NN recall is only ~0.37 even on a topic-matched slice. So
the OA-only frontier is placed not by a direct match but by (a) its topically-nearest wiki concept
as a soft bridge (cross-corpus NN stays topical post-centering — KL-ONE→its paper) and (b)
propagation through OA citations to anchored neighbors. **Binding constraints: bridge coverage +
whether the OA frontier inherits pedagogy via propagation — untested.** Caveat: frontier placement
leans partly on OA citations (recency-biased — the thing wiki was meant to fix), so best-of-both
would be *partial* (clean wiki-pedagogy for canonical work, blended for the frontier; likely OK
since frontier work *is* recent).

**Test that decides it:** build the NLP joint graph (wiki-NLP links + OA-NLP citations +
high-confidence cosine bridges), run the curriculum, and check whether OA-only frontier works land
near their prerequisite wiki concepts. That experiment — not the dictionary precision — answers
whether the dream is realizable.

#### Status / working notes (2026-05-30): joint graph RUN — partial best-of-both

Built and ran end-to-end. Results doc: **`docs/wikipedia-openalex-joint-curriculum.md`**; script
`artifacts_scripts/wiki/joint_graph_curriculum.py` (+ one-time OA-slice prep
`artifacts_scripts/wiki/_prep_oa_nlp_cache.py`).

- **Setup.** Joint graph over 12k wiki-NLP (bg-norm top-k of the cached
  `curriculum_subgraph_n40000.npz`) ∪ 12k OA-NLP (top-cosine T10181 articles, intra-set citation
  graph from `referenced_works`, 25k edges). bt_cos bridges (per-slice mean-matched cosine,
  round-trip cos), raw AND CSLS, τ∈{0.5,0.6,0.7}. 24k nodes, ~1.6M edges. Pipeline: hub-suppressed
  seeded PageRank → link-LPP (k=64) on the joint affinity → bisecting-KMeans (14 topics) →
  frontier-adjacency order. ~30 s/run on CPU.
- **bt_cos clears the coverage bar.** CSLS bt_cos τ=0.5 → wiki/OA coverage 0.43/0.46, above strict
  mutual-NN (0.32) and the ~0.37 baseline; raw cosine weak (0.25/0.29). Coverage is NOT the
  bottleneck.
- **Where an OA-only paper survives the frontier cut, it lands correctly.** Clean win: niche OA
  parsing papers (Korean parser, PP-attachment, shallow parsing) cluster WITH wiki *Parsing / Formal
  grammar / Formal language / Syntax* and order at the frontier — pedagogy inherited through the
  joint graph.
- **BUT the binding constraint is OA mass starvation, not bridges.** Only ~13% of top-k OA nodes
  (CSLS; ~7% raw) reach a NAMED frontier topic — the rest sink into the dendrogram's `-1` unassigned
  mass (wiki: ~43%). The dense wiki link backbone pools PageRank mass; the sparse intra-OA citation
  graph can't compete, so bg-norm + frontier-mass-cut discard the OA long tail before clustering.
  Enriches the *canonical* frontier, loses the niche payload the dream wanted. No recency-bias
  takeover (sparse OA graph too weak to dominate — flip side of the same problem).
- **Next (live follow-up, OA side):** densify the OA backbone (citation-neighborhood expansion or an
  OA similarity-kNN graph, à la `seed_from_gap --similarity`) and/or boost OA seed mass so the OA
  frontier survives the mass cut. The bridge dictionary is already good enough; the fix is graph
  density on the OA side.
