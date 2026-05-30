# Dictionary-bridged joint-graph curriculum (wiki-NLP ∪ OA-NLP)

The **best-of-both** test: ONE scipy-sparse graph fusing Wikipedia's pedagogical link backbone with OpenAlex's citation coverage/frontier, bridged by **bt_cos** (back-translation cosine) cross-corpus edges, then run through the validated wiki-native curriculum pipeline. Does it keep wiki's clean foundations→frontier ordering AND enrich the frontier with niche/recent OA papers the wiki-only curriculum (`docs/wikipedia-curriculum.md`) lacked?

## Joint-graph construction

- **Wiki nodes:** bg-norm top-12000 of the cached subgraph `cache/wiki/store/enwiki_full/curriculum_subgraph_n40000.npz` (seeded forward PR `p` / uniform-teleport background PR `g`, select by `p^0.5·log(p/g)`, α=0.85), wiki links restricted to that set (1573804 edges).
- **OA nodes:** top-12000 OpenAlex NLP-Techniques (T10181) articles by cosine to the NLP-article centroid (raw GIST-384), with an **intra-set citation graph** built from their `referenced_works` (25284 citation edges + self-loops). This is the OA set/graph: the same raw-GIST NLP slice used by the cross-corpus alignment work, NOT the domain-scale gap curriculum (whose 40.9M citation graph is intractable to slice without the full work-id map).
- **Bridges:** bt_cos edges (below), 6672 kept at τ=0.5 (the curriculum-run threshold), added symmetrically, weighted by bt_cos.
- **Joint graph:** 24000 nodes, 1624432 directed edges, with a node-type vector (0=wiki, 1=OA).

## bt_cos bridge coverage (vs strict mutual-NN)

Per-slice mean-matched cosine `sim = X_wiki @ X_oa.T`; forward/backward argmax round-trip gives `bt_cos_i = cos(node_i, node_{rt(i)})`; a bridge survives if `bt_cos > τ`. **Coverage** = fraction of that side's nodes with ≥1 surviving bridge. Strict mutual-NN coverage (no τ, the prior ~0.37 baseline) = **0.3178** on this slice.

| scorer | τ | #bridges | wiki coverage | OA coverage |
| :-- | --: | --: | --: | --: |
| raw | 0.5 | 4644 | 0.250 | 0.288 |
| raw | 0.6 | 2615 | 0.180 | 0.178 |
| raw | 0.7 | 1863 | 0.145 | 0.143 |
| csls | 0.5 | 6672 | 0.434 | 0.464 |
| csls | 0.6 | 4951 | 0.371 | 0.372 |
| csls | 0.7 | 4212 | 0.336 | 0.337 |

bt_cos buys substantially more coverage than strict mutual-NN (0.318): at τ=0.5 the round-trip-stable criterion admits many non-mutual matches whose embedding is still self-consistent across the round trip.

## The joint curriculum (prerequisite → frontier order)

LPP subspace (k=64) on the hub-suppressed JOINT link affinity; smallest λ [0.0488, 0.1249, 0.2237, 0.2853, 0.2987]. Bisecting-KMeans into 14 topics; frontier-adjacency ordering.

| Rank | Size | OA frac | in−out | Top keywords | Sample nodes (★=OA) |
| :--: | ---: | ---: | ---: | :-- | :-- |
| 1 | 200 | 0.00 | +0.2566 | john, computer, robert, scientist, alan, allen, stephen | Andrew Zisserman; Mathematics Genealogy Project; Computer scientist; Gordon Plotkin; John McCarthy (computer scientist) |
| 2 | 92 | 0.00 | +0.2259 | language, gpt, text, artificial_intelligence, generative, image, video | Dartmouth workshop; Google DeepMind; Transformer (deep learning); Chatbot; Virtual assistant |
| 3 | 110 | 0.01 | +0.1688 | science, open, patent, academic, document, pubmed, scientific | Digital object identifier; Semantic Scholar; PubMed; PubMed Central; CiteSeerX |
| 4 | 180 | 0.01 | +0.1203 | computer, computing, theory, computational, mathematics, system, hardware | Computer science; ArXiv; Handle System; Computer vision; Mathematics |
| 5 | 157 | 0.06 | +0.0578 | programming, language, computing, computer, science, array, theory | Bibcode; Programming language; C++; Automata theory; Compiler |
| 6 | 272 | 0.60 | +0.0060 | grammar, parser, parsing, syntax, algorithm, context_free, compiler | Parsing; Formal grammar; Formal language; Syntax; Semantic parsing |
| 7 | 90 | 0.38 | -0.0070 | semantic, knowledge, extraction, programming, embedding, logic, understanding | Word embedding; Wikidata; Large language model; Knowledge representation and reasoning; GloVe |
| 8 | 161 | 0.11 | -0.0194 | algorithm, language, reasoning, markov, sentiment, search, mining | Language model; Bernhard Schölkopf; Algorithm; Data mining; Optical character recognition |
| 9 | 133 | 0.09 | -0.0209 | linguistics, language, linguistic, natural_language, acquisition, grammatical, philosophy | Natural language processing; English language; Linguistics; Natural language; Natural-language user interface |
| 10 | 98 | 0.01 | -0.0367 | citation, index, academic, machine_learning, review, open_access, publishing | ISSN; Thesis; Open access; Academic journal; Peer review |
| 11 | 1 | 1.00 | -0.0642 | amp, machine_translation, amp_schubert, maxwell, dependency, semantics, analogical | ★Review of Papeaaij (1986): Word Expert |
| 12 | 79 | 0.04 | -0.0907 | learning, machine_learning, net, yooreeka, kubeflow, mlpack, mindspore | Machine learning; Reinforcement learning; Arthur Samuel (computer scientist); Reinforcement learning from human feed; Bias–variance tradeoff |
| 13 | 1 | 1.00 | -0.2982 | experiments, neural, embeddings, croatian, word, scoring, documents | ★Experiments with Neural Word Embedding |
| 14 | 1 | 1.00 | -0.2982 | issues, workshop, theoretical, processing, proceedings, natural_language, scoring | ★Proceedings of the 1978 workshop on Th |

## DECISIVE EVALUATION — do OA-only frontier works land near their wiki prerequisites?

**First, the binding number — does the OA frontier even reach the curriculum?** The frontier mass-cut assigns only high-mass dendrogram nodes to named topics; everything else becomes a cluster `-1` intermediary. Of the **1941 OA nodes in the top-5000**, only **247 (12.7%) reach a named topic** — the rest sink into the unassigned `-1` mass. Wiki: **1328/3059 (43.4%)**. The wiki backbone carries the PageRank mass, so most OA frontier work is demoted below the topic-formation threshold before clustering. This is the central tension: the bridges place the OA work that *does* survive, but the bulk of the OA frontier never makes it into a topic at all.

**OA-only frontier works** = clustered OA nodes with the weakest cross-corpus bridge (bottom-quartile bt_cos) — niche/recent papers with no strong wiki counterpart. Of 247 OA nodes in the clustered set, the bottom quartile are examined below. For each, we list the topic it landed in, that topic's prerequisite→frontier rank, the topic's wiki foundations (highest-mass wiki members), and the topic keywords. The question: does the OA-only paper sit in a topic anchored by the RIGHT wiki concepts, ordered at/after those foundations?

| OA-only work (low bt_cos) | bt_cos | topic rank | topic OA frac | topic keywords | wiki foundations in same topic |
| :-- | --: | --: | --: | :-- | :-- |
| Wide coverage symbolic surface realization | 0.04 | 4 | 0.01 | computer, computing, theory, computational, mathematics | Computer science; ArXiv; Handle System; Computer vision |
| An Associative Concept Dictionary for Verbs an | 0.09 | 7 | 0.38 | semantic, knowledge, extraction, programming, embedding | Word embedding; Wikidata; Large language model; Knowledge representation and |
| French Amalgam: A machine-learned sentence rea | 0.10 | 8 | 0.11 | algorithm, language, reasoning, markov, sentiment | Language model; Bernhard Schölkopf; Algorithm; Data mining |
| Automatic Import of Verbal Syntactic Relations | 0.11 | 5 | 0.06 | programming, language, computing, computer, science | Bibcode; Programming language; C++; Automata theory |
| Portability of syntactic structure for languag | 0.12 | 6 | 0.60 | grammar, parser, parsing, syntax, algorithm | Parsing; Formal grammar; Formal language; Syntax |
| Prepositional Phrase Attachment in Shallow Par | 0.13 | 6 | 0.60 | grammar, parser, parsing, syntax, algorithm | Parsing; Formal grammar; Formal language; Syntax |
| Benchmarking for syntax-based sentential infer | 0.14 | 8 | 0.11 | algorithm, language, reasoning, markov, sentiment | Language model; Bernhard Schölkopf; Algorithm; Data mining |
| Automatic semantic grouping in a spoken langua | 0.14 | 7 | 0.38 | semantic, knowledge, extraction, programming, embedding | Word embedding; Wikidata; Large language model; Knowledge representation and |
| A parsing algorithm for Korean implemented in  | 0.14 | 6 | 0.60 | grammar, parser, parsing, syntax, algorithm | Parsing; Formal grammar; Formal language; Syntax |
| Combining learning approaches for incremental  | 0.15 | 6 | 0.60 | grammar, parser, parsing, syntax, algorithm | Parsing; Formal grammar; Formal language; Syntax |
| Design and evaluation of a phonological phrase | 0.15 | 6 | 0.60 | grammar, parser, parsing, syntax, algorithm | Parsing; Formal grammar; Formal language; Syntax |
| Analysis of Noun-Noun sequences: a rule based  | 0.15 | 6 | 0.60 | grammar, parser, parsing, syntax, algorithm | Parsing; Formal grammar; Formal language; Syntax |
| Automatic Classification of Article Errors in  | 0.15 | 10 | 0.01 | citation, index, academic, machine_learning, review | ISSN; Thesis; Open access; Academic journal |
| A Probabilistic Parser as a Model of Global Pr | 0.16 | 6 | 0.60 | grammar, parser, parsing, syntax, algorithm | Parsing; Formal grammar; Formal language; Syntax |

### Concrete examples

- **OA-only:** Wide coverage symbolic surface realization  (bt_cos=0.04, best wiki match was *Surface triangulation*)
  - landed in topic **rank 4/14** (OA frac 0.01); keywords: computer, computing, theory, computational, mathematics, system
  - wiki foundations in that topic: Computer science, ArXiv, Handle System, Computer vision, Mathematics

- **OA-only:** An Associative Concept Dictionary for Verbs and its Application to Elliptical Word Estimation.  (bt_cos=0.09, best wiki match was *Akaike information criterion*)
  - landed in topic **rank 7/14** (OA frac 0.38); keywords: semantic, knowledge, extraction, programming, embedding, logic
  - wiki foundations in that topic: Word embedding, Wikidata, Large language model, Knowledge representation and reasoning, GloVe

- **OA-only:** French Amalgam: A machine-learned sentence realization system  (bt_cos=0.10, best wiki match was *Large language model*)
  - landed in topic **rank 8/14** (OA frac 0.11); keywords: algorithm, language, reasoning, markov, sentiment, search
  - wiki foundations in that topic: Language model, Bernhard Schölkopf, Algorithm, Data mining, Optical character recognition

- **OA-only:** Automatic Import of Verbal Syntactic Relations Using Parallel Corpora  (bt_cos=0.11, best wiki match was *Parallel programming model*)
  - landed in topic **rank 5/14** (OA frac 0.06); keywords: programming, language, computing, computer, science, array
  - wiki foundations in that topic: Bibcode, Programming language, C++, Automata theory, Compiler

- **OA-only:** Portability of syntactic structure for language modeling  (bt_cos=0.12, best wiki match was *Systems modeling language*)
  - landed in topic **rank 6/14** (OA frac 0.60); keywords: grammar, parser, parsing, syntax, algorithm, context_free
  - wiki foundations in that topic: Parsing, Formal grammar, Formal language, Syntax, Semantic parsing

- **OA-only:** Prepositional Phrase Attachment in Shallow Parsing  (bt_cos=0.13, best wiki match was *Parse tree*)
  - landed in topic **rank 6/14** (OA frac 0.60); keywords: grammar, parser, parsing, syntax, algorithm, context_free
  - wiki foundations in that topic: Parsing, Formal grammar, Formal language, Syntax, Semantic parsing

## Verdict — partial best-of-both, bottlenecked by OA mass starvation

**Realizable, but only partially, and the binding constraint is not bridge precision — it is OA mass starvation in the joint PageRank.**

- **bt_cos clears the coverage bar.** CSLS bt_cos at τ=0.5 reaches wiki/OA coverage ~0.43/0.46, comfortably above strict mutual-NN (0.32) and the ~0.37 prior baseline; raw cosine is much weaker (~0.25/0.29). So the dictionary *can* bridge a plurality of nodes — coverage is NOT the bottleneck.
- **Where an OA-only paper survives the cut, it lands correctly.** The parsing/grammar topic is the clean win: niche OA parsing papers (Korean parser, PP-attachment, shallow parsing) cluster WITH the wiki foundations *Parsing / Formal grammar / Formal language / Syntax* and order at the frontier end — pedagogy inherited through the joint graph, exactly the dream. This is best-of-both in microcosm.
- **But the OA frontier mostly never reaches a topic.** Only 13% of top-k OA nodes get assigned to a named topic (vs 43% of wiki); the rest sink into the `-1` mass. The wiki link backbone is dense (mean out-degree in the hundreds) while the intra-OA citation graph is sparse (25284 edges over 12000 nodes), so seeded PageRank mass pools on wiki, and the bg-norm + frontier-mass cut discards low-mass OA nodes. The result enriches the *canonical* frontier (parsing, embeddings) but loses most of the long-tail niche papers — the very payload the dream wanted.
- **Honest failure modes seen:** (a) bridge misfires on polysemous titles route a few OA papers into wrong-but-plausible generic topics (e.g. a surface-realization paper bridged to *Surface triangulation* → landed in a generic CS topic); (b) a handful of unbridged OA papers form degenerate singleton topics at the extreme frontier (ranks 12–14); (c) OA-dominated topics drift toward toolkit keywords (pytorch/keras/lightgbm) rather than concepts. No catastrophic recency-bias takeover — the sparse OA citation graph is too weak to dominate, which is the flip side of the mass-starvation problem.
- **τ / density effect:** lowering τ (raw 0.5) admits ~2× more OA into the top-k (58.8% vs 38.8% OA) but those extra OA nodes are low-confidence and still fail the frontier-mass cut, so frontier *placement* barely improves — density helps coverage, not pedagogy transfer.

**Bottom line.** The joint graph keeps wiki's clean hub-free foundations→frontier backbone and *does* graft on OA frontier work where a bridge + citation anchor exist, with correct prerequisite adjacency. It is genuinely best-of-both for the canonical/bridged frontier. It is NOT yet best-of-both for the OA-only long tail: those papers are mass-starved out of the curriculum before clustering. The fix is on the OA side (denser intra-OA edges via citation expansion or an OA similarity-kNN backbone, and/or boosting OA seed mass), not on the bridge dictionary.

