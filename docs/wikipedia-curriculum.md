# Wikipedia-link-graph curriculum (NLP) — cirrus-native

Curriculum built on the **cirrus-native Wikipedia link graph** parsed from the SAME CirrusSearch content dump (`enwiki-20251229`) as the **Wikipedia GIST embeddings** (store `cache/wiki/store/enwiki_full`). One internally coherent dump — no cross-dump page_id mismatch, no bookgen. Compared against the OpenAlex-citation curriculum `artifacts/curriculum/gap_curriculum_T11273/summary.md`.

## Setup

- **Subgraph selection:** top-40000 Wikipedia pages by cosine to the OpenAlex NLP-article centroid (streamed over all 7.1M embeddings; no dense index), unioned with their embedded cirrus out-neighbors.
- **Link subgraph (cirrus-native):** 477833 nodes, 68463618 directed article→article edges (mean out-degree 143.3, max 4811).
- **PageRank:** forward (mass→prerequisites), α=0.85, seeded at ['Natural language processing', 'Machine learning', 'Language model', 'Word embedding', 'Parsing', 'Computational linguistics'].
- **Clustering:** bisecting K-Means on GIST embeddings of the top-4000 pages by PR mass, cut to ~12 topics.
- **Ordering:** topics sorted by total PageRank mass (foundational → frontier).
- **Keywords:** log-odds + lattice phrase mining (same as engine `summarize_topics`), over page title + opening_text.


## Topics (prerequisite → frontier order)

| Rank | Size | PR mass | Top keywords | Sample pages |
| :--: | ---: | ---: | :-- | :-- |
| 1 | 233 | 0.0306 | algorithm, artificial_intelligence, computer_scientist, computer_science, known, alex_graves, systems, one | Computer science; Artificial intelligence; Algorithm; Artificial general intelligence; Alan Turing |
| 2 | 230 | 0.0298 | grammar, parser, language, computer_science, string, comparison, parsing, regular_expression_engines | Formal grammar; Programming language; Grammar checker; Formal language; Compiler |
| 3 | 259 | 0.0290 | software, system, systems, use, computer, design, information, programmer | Digital object identifier; Bibcode; Handle System; Wikidata; Information retrieval |
| 4 | 341 | 0.0283 | university, academic, published, scientific, science, publishing, books, articles | PubMed; ArXiv; OCLC; PubMed Central; CiteSeerX |
| 5 | 349 | 0.0278 | iso, standard, iso_iec, international, standards, published, organization, iso_iec_iso_iec | ISBN; ISSN; ISO 4; International Organization for Standardi; PDF |
| 6 | 1 | 0.0260 | machine_learning, unseen, minimisation, unsupervised, exploratory, probably, viewpoint, generalise | Machine learning |
| 7 | 1 | 0.0259 | broadly, natural_language_processing_nlp, closely, subfield, classification, associated, major, processing | Natural language processing |
| 8 | 1 | 0.0257 | sentence, constituents, speech, grammatical, computer_languages, comprehension, orationis, discussing | Parsing |
| 9 | 1 | 0.0256 | computational, modelling, interdisciplinary_field, neuroscience, anthropology, cognitive_psychology, among_others, cognitive_science | Computational linguistics |
| 10 | 319 | 0.0255 | computer, system, systems, network, computing, software, design, digital | Optical character recognition; Operating system; Database; Automatic identification and data captur; Facial recognition system |
| 11 | 1 | 0.0255 | superseded, gram, handwriting, scraped, induction, llms, route, transformers | Language model |
| 12 | 1 | 0.0254 | embedding, embeddings, vector, real, dimensionality, explainable, boost, sentiment | Word embedding |

## Topic detail

### [1] Topic (size=233, PR mass=0.0306)

**Keywords:** algorithm, artificial_intelligence, computer_scientist, computer_science, known, alex_graves, systems, one, theory, time

**Representative pages:**
- Computer science
- Artificial intelligence
- Algorithm
- Artificial general intelligence
- Alan Turing

### [2] Topic (size=230, PR mass=0.0298)

**Keywords:** grammar, parser, language, computer_science, string, comparison, parsing, regular_expression_engines, text, strings

**Representative pages:**
- Formal grammar
- Programming language
- Grammar checker
- Formal language
- Compiler

### [3] Topic (size=259, PR mass=0.0290)

**Keywords:** software, system, systems, use, computer, design, information, programmer, program, process

**Representative pages:**
- Digital object identifier
- Bibcode
- Handle System
- Wikidata
- Information retrieval

### [4] Topic (size=341, PR mass=0.0283)

**Keywords:** university, academic, published, scientific, science, publishing, books, articles, american, journals

**Representative pages:**
- PubMed
- ArXiv
- OCLC
- PubMed Central
- CiteSeerX

### [5] Topic (size=349, PR mass=0.0278)

**Keywords:** iso, standard, iso_iec, international, standards, published, organization, iso_iec_iso_iec, systems, international_standard

**Representative pages:**
- ISBN
- ISSN
- ISO 4
- International Organization for Standardization
- PDF

### [6] Topic (size=1, PR mass=0.0260)

**Keywords:** machine_learning, unseen, minimisation, unsupervised, exploratory, probably, viewpoint, generalise, analytics, subdiscipline

**Representative pages:**
- Machine learning

### [7] Topic (size=1, PR mass=0.0259)

**Keywords:** broadly, natural_language_processing_nlp, closely, subfield, classification, associated, major, processing, ipsn, secured

**Representative pages:**
- Natural language processing

### [8] Topic (size=1, PR mass=0.0257)

**Keywords:** sentence, constituents, speech, grammatical, computer_languages, comprehension, orationis, discussing, separation, pars

**Representative pages:**
- Parsing

### [9] Topic (size=1, PR mass=0.0256)

**Keywords:** computational, modelling, interdisciplinary_field, neuroscience, anthropology, cognitive_psychology, among_others, cognitive_science, questions, draws

**Representative pages:**
- Computational linguistics

### [10] Topic (size=319, PR mass=0.0255)

**Keywords:** computer, system, systems, network, computing, software, design, digital, may, hardware

**Representative pages:**
- Optical character recognition
- Operating system
- Database
- Automatic identification and data capture
- Facial recognition system

### [11] Topic (size=1, PR mass=0.0255)

**Keywords:** superseded, gram, handwriting, scraped, induction, llms, route, transformers, datasets, predominantly

**Representative pages:**
- Language model

### [12] Topic (size=1, PR mass=0.0254)

**Keywords:** embedding, embeddings, vector, real, dimensionality, explainable, boost, sentiment, mapping, reduction

**Representative pages:**
- Word embedding

## Verdict vs the OpenAlex-citation curriculum

**As-is, the Wikipedia link-graph curriculum is worse — and it fails in a classic,
diagnosable way: bibliographic-infrastructure hub contamination.**

Forward PageRank concentrates "prerequisite mass" on globally-popular *metadata*
pages, not conceptual foundations. Three of the top-five topics are pure
citation/identifier infrastructure:

- **Topic 3** — Digital object identifier, Bibcode, Handle System, Wikidata
- **Topic 4** — PubMed, ArXiv, OCLC, PubMed Central, CiteSeerX
- **Topic 5** — ISBN, ISSN, ISO 4, PDF

These pages are linked from nearly every article via citation templates and
identifier infoboxes, so they have enormous global in-degree (max out-degree in
the subgraph is 4811) but zero pedagogical content. They **outrank the actual
NLP foundations**, and the six seed pages (Machine learning, NLP, Parsing,
Computational linguistics, Language model, Word embedding) each collapsed into a
**size-1 singleton** cluster ranking *below* the infrastructure hubs. Frontier
concepts (Transformers, BERT, attention, neural MT) barely surface — only as
keywords under the singleton *Language model* topic.

The OpenAlex-citation curriculum (`gap_curriculum_T11273`) has none of this: 15
clusters, **0 singletons**, sizes 68–417, content-rich topics anchored by real
foundational works — *Parallel Distributed Processing*, Bishop's *Neural Networks
for Pattern Recognition*, *Latent Semantic Analysis*, Bengio's *A Neural
Probabilistic Language Model*, *Matrix Analysis*, *Generalized Iterative Scaling*
— with a clean progression into graph/network embeddings. Academic citations
point to **papers**, so PageRank rewards seminal work; Wikipedia links point to
**identifier registries**, so PageRank rewards bureaucracy.

**There is real signal under the noise**, though: topics 1–2 (Computer
science / AI / Algorithm / Turing; Formal grammar / Programming language /
Compiler) are sensible foundations. The link graph is usable *if the hubs are
suppressed.*

### Salvage paths (backlogged)

1. **Hub suppression** — downweight or drop edges into very-high-global-in-degree
   pages (the infra hubs sit at the extreme tail of the in-degree distribution);
   in-degree-normalized edge weights, or drop the top-N global hubs.
2. **Infrastructure blocklist / category filter** — exclude pages in
   bibliographic-identifier / database / standards-org categories. The cirrus
   dump already carries each page's `category` list, so this is a one-pass filter
   over data we have.
3. **Background-normalized PageRank** — this is exactly what the
   `p^a·log(p/g)` background normalization already built for the OpenAlex
   curriculum does: demote nodes with high *background* (un-seeded) mass. DOI /
   ISBN / PubMed have huge background mass and would be demoted automatically.
   The most principled fix, and it reuses existing machinery.

### Reproducing / iterating

`title_maps.parquet` (322 MB) and `curriculum_subgraph_n40000.npz` (956 MB) are
cached under `cache/wiki/store/enwiki_full/`, so re-running with
`--reuse-subgraph` skips the 43 GB dump entirely — seed/α/clustering/hub-filter
tuning is seconds, not 50 minutes.

