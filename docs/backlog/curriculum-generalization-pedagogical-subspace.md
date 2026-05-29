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
