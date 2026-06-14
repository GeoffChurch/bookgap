# Wikipedia-link-graph curriculum (NLP) — cirrus-native

Curriculum built on the **cirrus-native Wikipedia link graph** parsed from the SAME CirrusSearch content dump (`enwiki-20251229`) as the **Wikipedia GIST embeddings** (store `cache/wiki/store/enwiki_full`). One internally coherent dump — no cross-dump page_id mismatch, no bookgen. Compared against the OpenAlex-citation curriculum `artifacts/curriculum/gap_curriculum_T11273/summary.md`.

## Setup

- **Subgraph selection:** top-40000 Wikipedia pages by cosine to the OpenAlex NLP-article centroid (streamed over all 7.1M embeddings; no dense index), unioned with their embedded cirrus out-neighbors.
- **Link subgraph (cirrus-native):** 477833 nodes, 68463618 directed article→article edges (mean out-degree 143.3, max 4811).
- **PageRank:** forward (mass→prerequisites), α=0.85, seeded at ['Natural language processing', 'Machine learning', 'Language model', 'Word embedding', 'Parsing', 'Computational linguistics'].
- **Background normalization:** top-k SELECTION ranked by `p^a · log((p+ε)/(g+ε))` (a=0.5, α_bg=0.85, ε=1e-12), where p = seeded forward PR and g = uniform-teleport background PR on the same adjacency. Demotes globally-popular hubs. Topic ordering uses the mean lift score; cluster weights stay raw PR mass.
- **Clustering:** bisecting K-Means on GIST embeddings of the top-4000 pages by background-normalized score, cut to ~12 topics.
- **Ordering (directional, bg-normalized):** inter-topic dependency matrix `compute_frontier_adjacency` over the RAW forward-PR flow, but the in-flow to each topic is background-normalized: `inflow_fg^a · log((inflow_fg+ε)/(inflow_bg+ε)) − outflow_fg` (a=0.5, ε=1e-12), where inflow_bg is the same absorbing in-flow under a UNIFORM (un-seeded) global source. Demotes globally-popular hub topics (the topic-level analog of the node-selection bg-norm). Decoupled from the bg-norm SELECTION.
- **Keywords:** log-odds + lattice phrase mining (same as engine `summarize_topics`), over page titles.


## Topics (prerequisite → frontier order)

| Rank | Size | lift | bgnorm-in−out | Top keywords | Sample pages |
| :--: | ---: | ---: | ---: | :-- | :-- |
| 1 | 136 | 0.0467 | +2.8778 | gpt, openai, deep_learning, chatbot, text, image, sentiment, multimodal | Transformer (deep learning); Seq2seq; Neural machine translation; Word2vec; Deep learning |
| 2 | 479 | 0.0193 | +2.4354 | john, ibm, conference, arxiv, turing, fei, quoc, scientist | Joseph Weizenbaum; Restricted Boltzmann machine; Yoshua Bengio; International Conference on Learning Rep; ECML PKDD |
| 3 | 304 | 0.0238 | +2.3217 | programming, parser, compiler, translation, text, computer, machine, recognition | Natural-language user interface; Machine translation; Transfer-based machine translation; Rule-based machine translation; Concordancer |
| 4 | 312 | 0.0167 | +1.9995 | software, computer, computing, system, security, information, graphics, development | Computer-assisted reviewing; Automated essay scoring; Automatic identification and data captur; Human-in-the-loop; Robot control |
| 5 | 461 | 0.0115 | +1.5858 | logic, mathematics, theory, formal, reasoning, type, set, recursion | Chomsky normal form; Simple precedence grammar; Reasoning model; Semi-Thue system; String operations |
| 6 | 357 | 0.0148 | +1.4627 | statistics, statistical, regression, linear, variance, correlation, factor, learning | Topic model; Compound-term processing; Latent Dirichlet allocation; Overfitting; Bias–variance tradeoff |
| 7 | 1 | 1.0428 | +0.4406 | natural, processing, generation, making, ashish, checker, lotfi, continuous | Natural language processing |
| 8 | 1 | 1.1627 | +0.3383 | linguistics, computational, generation, making, ashish, checker, lotfi, continuous | Computational linguistics |
| 9 | 1 | 0.9714 | +0.3340 | machine_learning, generation, making, ashish, checker, lotfi, continuous, integrated | Machine learning |
| 10 | 1 | 1.1884 | +0.2795 | parsing, generation, making, ashish, checker, lotfi, continuous, integrated | Parsing |
| 11 | 1 | 1.3442 | +0.2502 | embedding, word, generation, making, ashish, checker, lotfi, continuous | Word embedding |
| 12 | 1 | 1.2491 | +0.1853 | language, generation, making, ashish, checker, lotfi, continuous, integrated | Language model |

## Topic detail

### [1] Topic (size=136, lift=0.0467)

**Keywords:** gpt, openai, deep_learning, chatbot, text, image, sentiment, multimodal, udio, apertium

**Representative pages:**
- Transformer (deep learning)
- Seq2seq
- Neural machine translation
- Word2vec
- Deep learning

### [2] Topic (size=479, lift=0.0193)

**Keywords:** john, ibm, conference, arxiv, turing, fei, quoc, scientist, alphafold, andrew

**Representative pages:**
- Joseph Weizenbaum
- Restricted Boltzmann machine
- Yoshua Bengio
- International Conference on Learning Representations
- ECML PKDD

### [3] Topic (size=304, lift=0.0238)

**Keywords:** programming, parser, compiler, translation, text, computer, machine, recognition, expression, speech

**Representative pages:**
- Natural-language user interface
- Machine translation
- Transfer-based machine translation
- Rule-based machine translation
- Concordancer

### [4] Topic (size=312, lift=0.0167)

**Keywords:** software, computer, computing, system, security, information, graphics, development, network, multiprocessing

**Representative pages:**
- Computer-assisted reviewing
- Automated essay scoring
- Automatic identification and data capture
- Human-in-the-loop
- Robot control

### [5] Topic (size=461, lift=0.0115)

**Keywords:** logic, mathematics, theory, formal, reasoning, type, set, recursion, logical, system

**Representative pages:**
- Chomsky normal form
- Simple precedence grammar
- Reasoning model
- Semi-Thue system
- String operations

### [6] Topic (size=357, lift=0.0148)

**Keywords:** statistics, statistical, regression, linear, variance, correlation, factor, learning, biology, smoothing

**Representative pages:**
- Topic model
- Compound-term processing
- Latent Dirichlet allocation
- Overfitting
- Bias–variance tradeoff

### [7] Topic (size=1, lift=1.0428)

**Keywords:** natural, processing, generation, making, ashish, checker, lotfi, continuous, integrated, manning

**Representative pages:**
- Natural language processing

### [8] Topic (size=1, lift=1.1627)

**Keywords:** linguistics, computational, generation, making, ashish, checker, lotfi, continuous, integrated, manning

**Representative pages:**
- Computational linguistics

### [9] Topic (size=1, lift=0.9714)

**Keywords:** machine_learning, generation, making, ashish, checker, lotfi, continuous, integrated, manning, biology

**Representative pages:**
- Machine learning

### [10] Topic (size=1, lift=1.1884)

**Keywords:** parsing, generation, making, ashish, checker, lotfi, continuous, integrated, manning, biology

**Representative pages:**
- Parsing

### [11] Topic (size=1, lift=1.3442)

**Keywords:** embedding, word, generation, making, ashish, checker, lotfi, continuous, integrated, manning

**Representative pages:**
- Word embedding

### [12] Topic (size=1, lift=1.2491)

**Keywords:** language, generation, making, ashish, checker, lotfi, continuous, integrated, manning, biology

**Representative pages:**
- Language model

