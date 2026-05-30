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
