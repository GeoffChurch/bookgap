# Dictionary-bridged joint-graph curriculum (wiki-NLP ∪ OA-NLP)

The **best-of-both** test: ONE scipy-sparse graph fusing Wikipedia's pedagogical link backbone with OpenAlex's citation coverage/frontier, bridged by **bt_cos** (back-translation cosine) cross-corpus edges, then run through the validated wiki-native curriculum pipeline. Does it keep wiki's clean foundations→frontier ordering AND enrich the frontier with niche/recent OA papers the wiki-only curriculum (`docs/wikipedia-curriculum.md`) lacked?

> **v1 → v2.** The first build (below, committed `7b3429f`) found only ~13% of top-k OA nodes reached a named topic vs ~43% of wiki — **OA mass starvation**. The [**v2 section**](#v2-oa-density--mass-rebalancing) at the end fixes this with OA citation-neighborhood expansion + per-corpus pool selection; jump there for the current state.

## v1 — Joint-graph construction

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

---

## v2: OA density + mass rebalancing

v2 attacks the v1 bottleneck — OA mass starvation — with two structural knobs in `artifacts_scripts/wiki/joint_graph_curriculum.py`. All v2 runs use the **CSLS bt_cos bridges at τ=0.5** (the v1 winner). The v1-equivalent baseline below is the committed v1 pipeline re-run under the CSLS bridges with every v2 knob OFF (it reproduces v1 exactly: OA 247/1941 = 12.7%, 25,284 OA edges).

### Fix (1) — OA citation-neighborhood expansion (`--oa-expand`, default on)

v1 built the OA citation graph over edges *within the top-12k cosine slice only*, but most NLP-paper citations point outside it, leaving the OA side artificially sparse (~2 edges/node). v2 instead takes the top-12k cosine works as **seeds**, then expands the node set to the **1-hop citation neighborhood within the full 87,408-work NLP slice**: every seed-cited work that is itself present in the 87k is added as a node, and the citation graph is built over the union (an edge survives whenever both endpoints are in the node set, so expansion nodes carry real degree, not just inbound citations).

| metric | v1 (within top-12k) | v2 (1-hop expanded) |
| :-- | --: | --: |
| OA nodes | 12,000 | **23,214** (12,000 seeds + 11,214 cited) |
| OA citation edges | 25,284 | **147,995** |
| OA edges/node | **2.11** | **6.38** (3.0×) |
| joint-graph nodes | 24,000 | **35,214** |
| joint-graph directed edges | 1,624,432 | 1,766,299 |

The node set stays tractable (~35k joint) and OA citation degree triples — undoing the artificial sparsity, exactly as diagnosed. Bridge coverage also rises on the denser slice: CSLS τ=0.5 wiki/OA coverage **0.53 / 0.40** (v1 0.43 / 0.46), strict mutual-NN 0.39 (v1 0.32).

### Fix (3) — mass rebalancing

The diagnosis is that PageRank stationary mass tracks in-degree, so the dense wiki link backbone hoards mass: even with a 50/50 teleport split, **the median wiki node carries ~120× the mass of the median OA node** (wiki amplifies the teleport mass through its dense links; the sparse OA citation graph barely propagates it). The dendrogram mass-cut in `cluster_embeddings` therefore demotes almost all OA to cluster `-1` *regardless of teleport*. Three levers are exposed:

- **`--per-corpus-select` (default ON, the reliable lever):** z-normalize the bg-norm selection score *within each corpus* before the top-k clustering-**pool** cut, so OA's presence in the pool tracks "high for its own corpus" rather than its (far smaller) absolute mass. This raises the OA share of the 5,000-node clustering pool from 38.8% (v1) to 65.2%, putting **3,260 OA into the pool** (v1: 1,941).
- **`--teleport-split` (default OFF, `-1`):** split the PageRank teleport/seed mass `wiki:OA` across corpora. Found to be an **unstable lever** — the dense wiki backbone re-concentrates the mass regardless, and the dendrogram cut on the bimodal mass distribution responds chaotically (a sweep over splits 0.5→0.1 gave OA coverage 16.5%, 8.5%, 59.3%, 13.5%, 7.2% — non-monotone, driven by whether one giant OA cluster happens to form). Kept available, defaulted off.
- **`--mass-rebalance count` (default OFF, `none`):** rescale each corpus's masses to its node share *for the dendrogram cut*. Same instability — it spawns degenerate OA singletons (7 vs 3) and *lowers* clean coverage to 12.1%. Defaulted off.

**Rebalancing actually used in the default v2 run: `--oa-expand` + `--per-corpus-select` (both on), mass-cut levers off.** The reliable fix is at the **pool-selection** stage (per-corpus z-norm), not the dendrogram mass-cut.

### The dual evaluation

**(1) OA coverage UP.** Re-run on the rebalanced, OA-denser joint graph:

| metric | v1 (CSLS, knobs off) | v2 (expand + per-corpus-select) |
| :-- | --: | --: |
| OA nodes in clustering pool | 1,941 | 3,260 |
| **OA reaching a named topic** | **247 (12.7%)** | **644 (19.8%)** |
| — of those, in non-singleton topics | 244 | 641 |
| — of those, in **wiki-anchored** (≥3 wiki) topics | 244 | **641** |
| wiki reaching a named topic | 1,328 (43.4%) | 778 (44.7%) |
| singleton topics | 3 | 3 |

OA coverage rose from **12.7% → 19.8%**, and in *absolute* terms the curriculum now carries **644 OA papers vs 247 — 2.6× more OA frontier** — with **641 of the 644 landing in wiki-anchored topics** (genuine best-of-both, not isolated OA blobs). This is a large improvement but stops short of full parity with wiki's ~44%: the OA long tail that has neither a strong bridge nor a citation anchor in the pool still clusters into low-mass dendrogram subtrees and is cut. Pushing the rate further (the teleport/mass-rebalance levers) reintroduces takeover/degeneracy rather than parity — so 19.8% with 641 wiki-anchored is the honest **sweet spot**, not a hard ceiling on the idea.

**(2) Pedagogy still intact.** The v1 parsing win survives and is enriched. v2 topic **rank 7** (`parser, parsing, dependency, parse, semantic, tree`, 66% OA) carries the wiki foundations **Parsing / Semantic parsing / Syntactic parsing**, and OA-only parsing papers (e.g. *Forest-guided Supertagger Training*, bt_cos 0.12) land there. The new dominant OA topic **rank 1** (`machine_translation`, 94% OA) is OA-dense because OA's NLP frontier genuinely *is* machine-translation-heavy — and it is correctly anchored by the wiki foundations **Machine translation / Interlingual MT / Neural MT**, with OA-only MT papers (*FBK MT systems*, *Word-to-Word Model of Translational Equivalence*, *phrase-based translation discrimination*, all bt_cos 0.07–0.18) attaching to it. The foundations→frontier ordering is clean: ranks 1–5 are wiki-foundation-heavy (scientists, CS, probability, open-science, computing), the NLP frontier (text/extraction, parsing, language models, NLP, linguistics, ML) sits at ranks 6–11, and `in−out` flow is positive at the foundations and negative at the frontier. The wiki backbone stays hub-free (bg-norm + hub-suppressed LPP as in v1).

**(3) No recency takeover.** Boosting OA did **not** reintroduce citation-recency bias or toolkit-keyword swamping. Every v2 topic keyword set is concept-driven (machine_translation, parsing, logic, probability, linguistics, machine_learning) — in fact *cleaner* than v1, which had a `yooreeka / kubeflow / mlpack / mindspore` toolkit topic; **v2 has no toolkit-keyword topic at all**. The 3 frontier singletons (ranks 12–14) are *old* papers (Croatian word-embedding experiments, Papeaaij 1986, the 1978 workshop), confirming the absence of a recency tilt. The danger of overcorrection lives entirely in the two defaulted-off mass-cut levers (teleport-split / mass-rebalance), which when pushed produce a single 1,000+-node OA-takeover topic — avoided by keeping them off.

### Verdict — best-of-both now holds at the scale that matters, with an honest residual

v2 **substantially closes the v1 bottleneck without overcorrecting.** The structural fix is two-part and the two parts matter for different reasons: **OA expansion** (Fix 1) restores OA's real citation degree (2.1 → 6.4 edges/node) so OA nodes are no longer isolated, and **per-corpus pool selection** (Fix 3) stops the global mass-cut from starving OA out of the clustering pool. Together they take OA from 247 → 644 papers in the curriculum (12.7% → 19.8% coverage), with 641/644 grafted onto wiki-anchored, correctly-ordered topics — and the wiki backbone's clean hub-free foundations→frontier structure is untouched. The parsing win generalizes, machine translation joins it as a second well-anchored OA-heavy topic, and there is no recency or toolkit takeover.

It is **not full parity with wiki** (~20% vs ~44%): the OA long tail with neither a bridge nor a citation anchor still falls below the dendrogram mass-cut. Crucially, the levers that *would* push the headline rate higher (teleport-split, per-corpus mass-rebalance for the cut) are **unstable on the bimodal joint mass distribution** — they buy a bigger number by spawning a degenerate OA-takeover cluster, which is worse pedagogy, so they are exposed but defaulted off. The remaining gap is therefore a real property of the dendrogram mass-cut on a two-corpus graph, not a knob left untuned; closing it cleanly would need a clustering/cut that is corpus-aware by construction (e.g. per-corpus dendrograms merged at the bridges) rather than a single global mass threshold — a v3 direction. As of v2, **best-of-both holds at scale for the bridged + citation-anchored OA frontier**, which is the majority of OA's actual NLP mass (MT, parsing, language models, IE), and the honest residual is the unanchored long tail.

---

## v3: smooth diffusion densification of the OA backbone

The v2 `--oa-expand` is a **hard, truncated diffusion**: a single (k=1, unweighted) hop that adds the 1-hop citation neighborhood as nodes and unit-weights the edges. v3 replaces that hard truncation with **Graph Diffusion Convolution** (GDC; Gasteiger/Klicpera et al., *"Diffusion Improves Graph Learning"*): a smooth, weighted, multi-hop re-wiring `S = Σ_k θ_k T^k` of the OA citation graph, with the diffusion **scale chosen** rather than fixed at one hop, then re-sparsified (top-`m`/row) + row-renormalized. Selected via `--oa-densify diffuse` (default stays `expand`; the expand path is **byte-identical** to v2 — re-verified at 644/3260 = 19.8%, 1,766,299 joint edges). Implementation: `artifacts_scripts/wiki/joint_graph_curriculum.py`.

**Node set + directedness.** Diffusion runs on the **same expanded node set** as the hard expansion (23,214 OA nodes), so the comparison is apples-to-apples — identical nodes, only the OA *edge structure* differs. The OA citation graph is directed (i→j ≡ "i cites j"); GDC's kernel is defined on an undirected graph (the heat/PPR kernel of a directed transition is not symmetric and the spectral/diffusion-map story assumes a symmetric operator), so we **symmetrize for the diffusion affinity**. But the curriculum's forward PageRank needs the citation *direction* (flow citer→cited so foundational works accrue mass → frontier ordering), so we **keep both**: the symmetric GDC affinity `S` (the densified backbone, weighted) is written into the joint graph's OA block AND the original directed citation skeleton is overlaid (unit-weighted). Diffuse symmetric, orient via the retained directed skeleton.

### Choosing the diffusion scale T — four criteria, all with interior optima

All four are computed and reported; the primary, used for the build, is **held-out-link** (`--select-t`, default). Heat kernel, grid `t ∈ {0.2, 0.5, 1, 2, 4, 8, 16, 32}`, on the real OA citation graph (147,995 directed edges over 23,214 nodes):

| criterion | picked `t` | curve shape (over-smoothing guard) |
| :-- | --: | :-- |
| **held-out-link** *(primary)* | **8** | interior peak — AUC 0.81 (t=0.2) → **0.836 (t=8)** → 0.79 (t=32) |
| modularity | 8 | interior peak — Q 0.05 (t=0.2) → **0.459 (t=8)** → 0.36 (t=32) |
| spectral (eigengap) | 4–16 | interior peak — gap rises to t=4–8, decays by t=32 |
| mass-match (eff. degree) | 16 | OA effective degree maxes below wiki's (OA is genuinely sparser), so it drifts to the high end of the grid |

**Agreement is strong:** three of four criteria land in `t ∈ [4, 16]`, with held-out-link and modularity **agreeing exactly at t=8**. Every criterion has a genuine **interior optimum** with a clear collapse at large `t` — the over-smoothing guard fires as designed (a naive "more diffusion = better" criterion would peg the grid maximum; none do). mass-match is the outlier only because OA's citation graph cannot reach wiki's link density at *any* scale, so "match wiki's degree" pushes it as high as the grid allows — a target-match limitation, not a disagreement about resolution. The held-out-link curve is the right primary: it is self-supervised, circularity-free (not a curriculum metric), and has the cleanest interior peak.

The held-out-link metric scores only held-out edges whose endpoints are **not 1-hop neighbors in the training graph** (recovering a 1-hop edge is trivial at any `t` and flattens the curve); ≥2-hop held-out edges are what make the AUC sensitive to the diffusion scale.

### top_m is the binding knob — and it must NOT match wiki's degree

The re-sparsification budget `top_m` (diffusion edges kept per OA node) turned out to dominate the outcome. Sweeping `top_m` at the held-out-link `t=8`:

| `top_m` | diffused edges/node | OA block edges/node | **OA reaching a named topic** | wiki-anchored | wiki coverage |
| --: | --: | --: | --: | --: | --: |
| 8 | 6.0 | 9.9 | 409 (12.2%) | 402 | 44.5% |
| 16 | 15.2 | — | 799 (24.3%) | 753 | 62.0% |
| **32 (default)** | **26.2** | **27.5** | **874 (27.0%)** | **770** | 68.4% |
| 131 (≈ wiki mean degree) | 109.9 | 110.1 | 209 (6.6%) | 205 | 27.6% |
| — *v2 hard expansion* | *6.4* | *6.4* | *644 (19.8%)* | *641* | *44.7%* |

Coverage is **non-monotone in `top_m`**, peaking near `top_m ≈ 32` then **collapsing** at `top_m = 131`. The natural first guess — set `top_m` to wiki's per-node link density (~131) so the two corpora "match" — is exactly wrong: at that budget the symmetric multi-hop diffusion floods the OA block with 110 edges/node, a handful of OA nodes become mega-hubs that hoard the PageRank mass, and the dendrogram mass-cut then promotes *fewer* subtrees (wiki coverage also craters, 68% → 28%, and the v2 MT win disappears). The default is therefore **`top_m = 32`** — a few× the hard-expansion density, the empirical sweet spot.

### Verdict — smooth diffusion BEATS the hard 1-hop expansion (at the right top_m)

At the held-out-link-chosen `t=8` with `top_m=32`, the v3 default carries **874 OA papers into the curriculum (27.0%) vs v2's 644 (19.8%)** — a 1.36× lift — with **770 landing in wiki-anchored topics** (v2: 641) and only **1 singleton** (v2: 3). Wiki coverage also rises (44.7% → 68.4%): the smooth multi-hop affinity raises *both* corpora's mass-holding so more of each survives the dendrogram cut, which is the structural fix v2's hard 1-hop could only partly deliver.

**The pedagogy survives the spot-check and is enriched.** The foundations→frontier ordering stays clean (ranks 1–5 wiki-foundation-heavy: probability/math/logic, publishing, linguistics, CS; the NLP frontier at ranks 6–12; `in−out` flow positive at foundations, negative at frontier). The **v2 wins persist:** the parsing topic (rank 7: `parser, grammar, parsing, compiler, context_free`, 50% OA) is anchored by *Parsing / Compiler / Formal grammar*, and machine translation (rank 6, 94% OA, anchored by corpus/statistics) plus a dedicated **language-modeling topic** (rank 9, 94% OA) anchored by *Katz's back-off model / Word n-gram language model / ACL / RAG / BLOOM* — OA-only LM papers (e.g. *A Neural Probabilistic Language Model*, *Improved backing-off for M-gram language models*) graft directly onto their right prerequisites. No recency or toolkit-keyword takeover.

**Honest residual.** The denser diffusion pulls a little bibliographic noise into a couple of topics' "wiki foundations" (ICanHazPDF, DBLP, Neo-Assyrian Text Corpus Project) — cosmetic, the topical members are correct. Coverage is still short of wiki parity in the OA-only deep tail, and the result is sensitive to `top_m` (the sweet spot is a band, not a single value). But the headline holds: **smooth GDC at a principled, criterion-chosen scale beats the crude hard 1-hop expansion on every metric — OA coverage, wiki-anchored count, singleton count — while preserving the foundations→frontier pedagogy**, and the four T-selection criteria substantively agree (three of four at `t ∈ [4,16]`, primary and modularity exactly at `t=8`) with all four exhibiting the required interior-optimum / no-over-smoothing behavior. The default is `--oa-densify diffuse --diffusion-kernel heat --select-t heldout-link --diffusion-top-m 32` (though `--oa-densify` itself still defaults to `expand` so the committed v2 pipeline is unchanged for other consumers).

