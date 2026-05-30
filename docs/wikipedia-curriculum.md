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

## Background-normalized PageRank

Salvage path 3 above, executed. The `--bg-norm` flag ranks the top-k SELECTION
by the background-normalized score `p^a · log((p+ε)/(g+ε))` instead of raw
seeded PR mass `p`, where `g` is a **uniform-teleport (no-seed) background
PageRank on the same adjacency** at the same damping (α_bg = α_fg = 0.85),
ε = 1e-12. This is exactly the scoring the OpenAlex curriculum added in commit
`dbef012` (`background_normalize` in `artifacts_scripts/curriculum/`), reused
verbatim here — `a` is the `--bg-alpha` exponent (canonical default 0.5 =
Fightin'-Words variance-normalized). Cluster weights stay raw PR mass; topic
ordering uses mean lift, so the demotion shows in the prerequisite ordering.

The background distribution `g` is the global PageRank with no seeds, so
DOI / ISBN / PubMed / ArXiv / ISO — linked from nearly every article via
citation templates — carry enormous `g`. Their seeded mass `p` is also large,
but `p/g ≈ 1` (the gap doesn't elevate them beyond their baseline popularity),
so `log(p/g)` is near zero and they sink. The genuine NLP foundations have
`p ≫ g` (the seeds specifically elevate them), so they survive.

Run (`--bg-norm --bg-alpha 0.5`, same subgraph / seeds / top-4000 / ~12 topics):

| Rank | Size | Mean lift | Top keywords | Sample pages |
| :--: | ---: | ---: | :-- | :-- |
| 1 | 1 | 1.344 | representation, embeddings, word_embedding, embedding | Word embedding |
| 2 | 1 | 1.249 | superseded, gram, recurrent_neural, induction | Language model |
| 3 | 1 | 1.188 | sentence, constituents, grammatical, parse | Parsing |
| 4 | 1 | 1.163 | computational, anthropology, cognitive_science, neuroscience | Computational linguistics |
| 5 | 1 | 1.043 | processing_tasks, natural_language_processing_nlp, subfield, knowledge_representation | Natural language processing |
| 6 | 1 | 0.971 | statistical_algorithms, optimisation, minimisation, generalise | Machine learning |
| 7 | 136 | 0.047 | gpt, large_language, openai, chatgpt, deep_learning | Transformer (deep learning); Seq2seq; Neural machine translation; Word2vec; Deep learning |
| 8 | 304 | 0.024 | language, text, parser, programming_language | Machine translation; Transfer-based MT; Rule-based MT; Concordancer |
| 9 | 479 | 0.019 | artificial_intelligence, machine_learning, deep_learning | Joseph Weizenbaum; Restricted Boltzmann machine; Yoshua Bengio; ICLR |
| 10 | 312 | 0.017 | software, system, hardware | Automated essay scoring; Human-in-the-loop; Robot control |
| 11 | 357 | 0.015 | statistical, statistics, population, sample, variance | Topic model; Latent Dirichlet allocation; Overfitting; Bias–variance tradeoff |
| 12 | 461 | 0.011 | mathematics, logic, set, function, computer_science | Chomsky normal form; Simple precedence grammar; Semi-Thue system; String operations |

### Verdict: it fixes it

**Background normalization fully removes the bibliographic-hub contamination.**
DOI, ISBN, PubMed, ArXiv, ISO, Wikidata, Bibcode, OCLC, ISSN, PDF, CiteSeerX,
PubMed Central, Handle System — every infrastructure hub that owned three of
the top-five topics in the raw-PR run — is gone from the topic table and from
every representative-page list. ISBN dropped out of the top-4000 selection
entirely; the rest survive only at rank 501+ (ArXiv 501, DOI 1241, PubMed 1358,
ISSN 3170), where they are absorbed as ordinary members of large clusters or
left unassigned (intermediary-only), never anchoring a topic.

Genuine foundations surface where the raw run had bureaucracy:

- **Statistics / probability** — topic 11 (statistics, population, sample,
  variance; Latent Dirichlet allocation, bias–variance tradeoff, overfitting).
- **Logic / set theory / formal grammar** — topic 12 (mathematics, logic, set,
  function; Chomsky normal form, Semi-Thue system, simple precedence grammar).
- **Linguistics** — the Computational linguistics and Parsing seeds, plus the
  rule-based / transfer-based MT cluster (topic 8).

**Frontier at sensible depth.** Transformer, Seq2seq, Neural machine
translation, Word2vec, Deep learning, GPT/ChatGPT cluster at topic 7 — *after*
the six foundational seeds and the statistics/logic prerequisites, which is the
correct prerequisite→frontier ordering.

**The seeds are still singletons** (topics 1–6, size 1 each). This is unchanged
from the raw run and is not a bg-norm artifact: the six seed pages carry by far
the largest seeded mass `p` (they *are* the personalization vector) and the
largest lift, so they rank first and cluster alone. Bg-norm reorders them by
lift but doesn't merge them — that would need a separate fix (e.g. excluding
seeds from the ranked selection, or down-weighting seed self-mass).

### `--bg-alpha` tuning (0.0 / 0.5 / 1.0)

`a = 0.5` (the OpenAlex canonical default) is the sweet spot — it neither
under- nor over-corrects:

- **a = 0.5 (variance-norm, default):** highest hub pushed to rank 501; zero
  hubs in any topic table or representative list; clean foundation/frontier
  split (table above).
- **a = 1.0 (pointwise KL):** still demotes the hubs out of the top-5 topics,
  but **slightly under-corrects** — ArXiv climbs back to rank 26 and resurfaces
  as the *top representative page* of an infrastructure-flavored cluster
  ("web, information, published, online, documents"). The pure-`log(p/g)`
  weighting (no `p^a` support term) over-rewards low-support nodes.
- **a = 0.0 (raw log-lift):** also removes all hubs from the topic tables, but
  the frontier becomes noisier — low-support oddities ("1 the Road",
  "Jabberwacky", "Kling AI", "DBRX") become top representatives because there is
  no `p^a` support term to anchor on well-attested pages.

So no tuning is needed: **keep the canonical `a = 0.5`.** a=1.0 lets one hub
leak back; a=0.0 trades hub-cleanliness for a flakier frontier.

Reproduce (seconds, reuses the cached subgraph):

```bash
conda run -p .conda python -m artifacts_scripts.wiki.wiki_curriculum_demo \
    --wiki-store cache/wiki/store/enwiki_full --reuse-subgraph \
    --ranking-top-k 4000 --desired-n-topics 12 --use-abstracts \
    --bg-norm --bg-alpha 0.5
```

## Directional ordering (raw-fg-PR frontier adjacency)

`--bg-norm` fixes *selection* (which nodes survive the top-k cut) but leaves a
*backwards ordering*: ordering the topics by mean bg-norm **lift** is a
*relevance* axis, so the six seeds — the most NLP-relevant pages by
construction — sort **first** and the genuine foundations (Statistics,
Math/Logic) sort **last**. That is the wrong way round for a curriculum.

The fix decouples the two axes. **SELECTION stays bg-norm** (`--bg-norm
--bg-alpha 0.5`, all six seeds retained). **ORDERING** switches to a
directional pairwise-PR-DAG built from the **raw forward PageRank flow**, via
`--order-by frontier`:

1. Cluster the bg-norm-selected top-4000 into 12 topics (unchanged).
2. Build a full-graph `leaf_to_cluster` over all 477 833 subgraph nodes
   (cluster id for the top-k, `-1` for everything else — those still act as
   intermediaries) and call the engine's
   `compute_frontier_adjacency(adj, frontier_nodes, leaf_to_cluster, α=0.85)`.
   It runs an **absorbing, batched** forward PageRank seeded from each topic's
   own nodes over the **raw** directed link adjacency (NOT the bg-norm scores),
   producing `adj[i,j]` = how much topic *i* depends on topic *j* (*j* is a
   prerequisite for *i*).
3. Order topics with `_compute_linearization_order_from_adjacency(adj)` —
   in-degree (depended-on) minus out-degree (depends-on), descending, so
   prerequisites come first.

No reimplementation: both functions are reused verbatim from
`artifacts_scripts/curriculum/{pagerank,build_curriculum}.py`. The only
adapter is the full-graph `leaf_to_cluster` and a `frontier_nodes =
arange(n_topics)` placeholder (its values are only used for `|T|`).

**Timing.** `compute_frontier_adjacency` ran in **35 s** on the 477 833-node /
68.5 M-edge subgraph (the 12 topic-batches converge together in 41 power
iterations). This is far below the ~28 min it took on the full OpenAlex graph,
so the coarser pairwise-forward-PR-mass-flow fallback is unnecessary here.

### New ordering (the inversion is fixed)

The `lift` column is the mean bg-norm selection score (what the **old** ordering
sorted on); `in−out` is the new directional sort key. They now point in
**opposite** directions — which is exactly the decoupling working.

| Rank | Size | lift | in−out | Sample pages |
| :--: | ---: | ---: | ---: | :-- |
| 1 | 479 | 0.019 | **+0.2932** | AI/ML hub — Yoshua Bengio, Restricted Boltzmann machine, Joseph Weizenbaum, ICLR |
| 2 | 312 | 0.017 | **+0.2450** | Software/systems — Human-in-the-loop, Robot control, automatic data capture |
| 3 | 304 | 0.024 | **+0.1560** | Language/text/parsers — Machine translation (rule/transfer-based), Concordancer |
| 4 | 357 | 0.015 | **+0.1089** | **Statistics/probability** — Latent Dirichlet allocation, Overfitting, Bias–variance tradeoff |
| 5 | 136 | 0.047 | **+0.1082** | Neural NLP — Transformer, Seq2seq, Word2vec, Deep learning, GPT/ChatGPT |
| 6 | 461 | 0.012 | **+0.1002** | **Math/logic/formal grammar** — Chomsky normal form, Semi-Thue system, set/function |
| 7 | 1 | 1.043 | −0.1242 | **Natural language processing** (seed) |
| 8 | 1 | 1.163 | −0.1408 | **Computational linguistics** (seed) |
| 9 | 1 | 1.188 | −0.1587 | **Parsing** (seed) |
| 10 | 1 | 0.971 | −0.1897 | **Machine learning** (seed) |
| 11 | 1 | 1.249 | −0.1985 | **Language model** (seed) |
| 12 | 1 | 1.344 | −0.1998 | **Word embedding** (seed) |

**Prediction confirmed.** All six seed singletons now land at the very end
(ranks 7–12), each with negative in−out — a target that depends on everything
has high out-degree and low in-degree, so it is ordered last. The
broadly-depended-on foundations rise to the front: Statistics/probability
(rank 4) and Math/Logic/formal-grammar (rank 6) carry positive in−out, exactly
inverting their last-place position under the lift ordering.

The `lift` column makes the inversion vivid: the six seeds have by far the
**highest** lift (≈1.0–1.34) — they *are* the most NLP-relevant pages, which is
precisely why the old lift ordering put them first — yet the directional in−out
score correctly ranks them **last**. `lift` (relevance) and `in−out`
(dependency direction) now disagree in sign across the whole table; that
disagreement is the decoupling.

### Verdict

This is a **sensible foundations→frontier curriculum** at the gross level: the
hard inversion (seeds first) is gone, and the seed *frontier* targets now
trail the *prerequisite* foundations. Keep `--order-by frontier` with bg-norm
selection as the recommended setting.

**Caveat — the in−out heuristic is still coarse.** Within the front block the
order is not pedagogically monotone: the broad AI/ML umbrella (rank 1) outranks
Statistics (rank 4) and Math/Logic (rank 6) purely because almost every NLP
topic links *into* broad-AI articles, inflating its in-degree — even though
Math and Statistics are the deeper prerequisites. This is the documented
weakness of `_compute_linearization_order_from_adjacency` (marked DEPRECATED
for "not distinguishing source from target"): a high-in-degree *hub* and a
high-in-degree *foundation* are indistinguishable to in−out.

The principled upgrade is the **voltage / potential method**
(`_compute_linearization_order_from_potentials` + `circuitrank.py`): clamp the
seeds to 0 V (target) and a foundational reference to 1 V, solve the graph
Laplacian, and order by node potential. That genuinely separates source from
target along the flow direction rather than counting raw degree. It needs a
source/target clamp definition for the Wikipedia graph (the OpenAlex pipeline
gets `point_potentials` from CircuitRank; the wiki demo would have to compute
them), so it is the right escalation **if** the front-block mis-ordering of
foundations vs. the AI hub turns out to matter. For fixing the original
seeds-first inversion, raw-fg-PR frontier adjacency is sufficient.

Reproduce (seconds, reuses the cached subgraph — never touches the 43 GB dump):

```bash
conda run -p .conda python -m artifacts_scripts.wiki.wiki_curriculum_demo \
    --wiki-store cache/wiki/store/enwiki_full --reuse-subgraph \
    --ranking-top-k 4000 --desired-n-topics 12 --use-abstracts \
    --bg-norm --order-by frontier
```

