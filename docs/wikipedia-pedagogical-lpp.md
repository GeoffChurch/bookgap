# Wikipedia pedagogical subspace via LPP on the link graph

Locality Preserving Projections (LPP) fit on the **Wikipedia link graph** as the
affinity (hub-suppressed), contrasted against the usual kNN-of-embeddings
affinity, to test whether reprojecting the curriculum into the link-LPP subspace
sharpens topic clustering. Reuses the cached subgraph
`cache/wiki/store/enwiki_full/curriculum_subgraph_n40000.npz` (477 833 nodes,
68 463 618 directed link edges, GIST-384 embeddings). CPU-only, no 43 GB dump,
no GPU. Code: `artifacts_scripts/wiki/wiki_pedagogical_lpp.py`. The full
machine-generated report (all eigenvalues, axis lists, both topic tables, the
auto metric summary) is at
`artifacts/curriculum/wiki_pedagogical_lpp/report.md`.

Motivation: `docs/backlog/curriculum-generalization-pedagogical-subspace.md`
("pedagogical subspace"). The Wikipedia link structure encodes "what you need to
read to understand X" more directly than GIST cosine geometry. LPP finds the
linear projection `P` (384×k) minimizing `Σ_ij W_ij ‖Pᵀxᵢ − Pᵀxⱼ‖²` — it pulls
**link-connected** articles together in the projected space. If the links carry
pedagogical signal beyond embedding similarity, reprojecting into the link-LPP
subspace should make the curriculum's topic clustering respect the human-curated
link structure better than raw GIST does.

## Method

- **LPP solver — reuses `bookgap.denoisers` numerics.** `LPPDenoiser` already
  solves the generalized eigenproblem `X_cᵀ L X_c a = λ X_cᵀ D X_c a` for the k
  smallest eigenvalues via the shared `_solve_eigh_robustly` (`scipy.linalg.eigh`
  on the two 384×384 reduced matrices), `X_c` = mean-centered (n×384) GIST,
  `L = D − W`, `D` = degree of the symmetrized affinity `W`. We call the same
  solver; the only addition is **discarding any trivial near-DC eigenvector**
  (`λ < 1e-6`). In practice the guard never fires (see eigenspectrum): the
  constant/DC component lives in the discarded mean of `X_c` and in the `XᵀDX`
  constraint, so the 384-d reduced problem produces no near-zero eigenvalue.
- **Affinity = the link graph (the novelty), hub-suppressed.** `W` = the
  symmetrized link adjacency, where each directed edge `i→j` is weighted by
  `1/log(indeg_j + e)` (j's global in-degree), then `W = (W_dir + W_dirᵀ)/2`;
  `D` is recomputed from this weighted `W`. **Hub suppression is essential** —
  without it the Laplacian inherits the DOI/ISBN/ISO bibliographic-hub pathology
  documented in `docs/wikipedia-curriculum.md` (those pages have in-degrees in
  the hundreds of thousands, so they would dominate `D` and the smoothness
  penalty). It works:

  | Highest-in-degree page | in-degree | edge weight `1/log(indeg+e)` |
  | :-- | ---: | ---: |
  | ISBN | 235 688 | 0.081 |
  | Digital object identifier | 131 711 | 0.085 |
  | Wayback Machine | 101 092 | 0.087 |
  | ISSN | 93 939 | 0.087 |
  | Semantic Scholar | 68 174 | 0.090 |
  | OCLC | 49 518 | 0.092 |

  The infra hubs carry ~0.08–0.09× the weight of a niche page (~1.0), so they no
  longer pin the Laplacian.

- **Clustering comparison — scoped to cluster quality.** The bg-norm-selected
  curriculum top-4000 (seeded forward PR / uniform-teleport background PR,
  `p^0.5·log(p/g)`, α=0.85 — *identical selection* to
  `docs/wikipedia-curriculum.md`) is clustered by the same bisecting-KMeans into
  ~12 topics, once in raw GIST and once in each LPP subspace (k ∈ {32, 64, 128}).
  PageRank selection and frontier ordering are graph-based and **unaffected** by
  the embedding reprojection — LPP changes only the clustering geometry and the
  representative/keyword selection — so the comparison is deliberately scoped to
  cluster quality. The load-bearing metric is **link-modularity Q** and
  **link-flow capture** on the hub-suppressed link subgraph restricted to the
  4000 clustered nodes: higher = the embedding clusters agree more with the
  human-curated link communities = the pedagogical signal we are testing for.

## Eigenspectrum

Smallest 8 LPP eigenvalues (ascending; smaller λ = a direction that better
preserves the affinity's locality):

| affinity | λ0 | λ1 | λ2 | λ3 | λ4 | λ5 | λ6 | λ7 |
| :-- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| link (hub-suppressed) | 0.188 | 0.240 | 0.252 | 0.296 | 0.304 | 0.322 | 0.326 | 0.335 |
| kNN-of-embeddings | 0.079 | 0.104 | 0.125 | 0.127 | 0.144 | 0.157 | 0.168 | 0.172 |

No trivial near-zero eigenvalue for either affinity (smallest link λ = 0.188,
kNN λ = 0.079), so the DC guard never fires. The **kNN spectrum sits ~2.4× lower
than the link spectrum** at every index — kNN-LPP directions are "cheaper" to
satisfy linearly because the GIST embedding geometry *already* aligns with its
own kNN graph (LPP-on-kNN is close to PCA-on-the-local-manifold). The link
spectrum is higher precisely because the link affinity is **not** already
encoded in the GIST geometry: forcing link-connected articles together costs
real variance. That gap is the first quantitative hint that the link graph
carries structure the embeddings do not.

## Experiment 1 — what the top link-LPP axes capture

Articles at the extreme +/− ends of the first four link-LPP directions
(`Y = X_c P`). Each axis is a coherent **topical / link-community contrast**, not
noise — the link graph partitions the 477k-node neighborhood into dense
link-communities and LPP reads them off as linear directions:

- **Axis 0** (λ=0.188): software/OS (Windows Vista features, Plug and play,
  Remote desktop, Microsoft) ↔ minority/regional languages & peoples (Barunggam
  language, Meitei, Paiwan people, Languages of Bihar).
- **Axis 1** (λ=0.240): language-family articles (Gum/Ram/Guang/Grass languages,
  Tamang) ↔ UK literature & culture/sport (The House of Fame, Theodore
  Dalrymple, Chariots of Fire, Sociology of sport).
- **Axis 2** (λ=0.252): French place-names in the Aisne (Moÿ-de-l'Aisne,
  Vic-sur-Aisne, …) ↔ Hindu/Indic philosophy (Brahman, Vyasa, Upanishads,
  Autobiography of a Yogi).
- **Axis 3** (λ=0.296): mathematics & computation (Computable function, Galois
  theory, Diophantine approximation, Quantum algorithm, Indian mathematics) ↔
  American TV/entertainment (America's Got Talent, The Voice, American Idol).

These are *domain/topic* axes driven by link density (place-name pages link to
nearby place-names, math pages link to math), with a clear pedagogical flavor on
the technical poles (axis 3's + end is essentially "theory of computation /
number theory"). They are not prerequisite-*depth* axes — LPP on an undirected,
symmetrized affinity captures *community*, not *direction* (direction is the job
of the separate frontier/voltage ordering in `docs/wikipedia-curriculum.md`,
which this experiment leaves untouched).

## Experiment 2 — reproject + re-cluster vs raw GIST

Same nodes, same clusterer, same target topic count. `link-Q` / `link-flow` are
Newman modularity and within-cluster edge fraction on the hub-suppressed link
subgraph over the 4000 clustered nodes (**the cross-space-comparable metrics,
same subgraph for every row**). `silhouette` is in the space the clustering ran
in (raw 384-d GIST vs k-d LPP) and so is *not* comparable across rows as an
absolute — reported per-space only. `jacc` = mean pairwise keyword Jaccard
distance; `#kw3+` = keywords shared by ≥3 topics.

| variant | k | #topics | #sing | silhouette | **link-Q** | **link-flow** | jacc-dist | #kw3+ |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| raw GIST (baseline) | — | 12 | 6 | −0.070 | 0.240 | 0.311 | 0.990 | 2 |
| LPP-link k=32 | 32 | 12 | 6 | −0.139 | 0.142 | 0.194 | 0.991 | 1 |
| **LPP-link k=64** | 64 | 12 | 6 | −0.100 | **0.295** | **0.387** | 0.990 | 1 |
| LPP-link k=128 | 128 | 12 | 6 | −0.075 | 0.225 | 0.301 | 0.988 | 3 |
| LPP-kNN k=64 | 64 | 12 | 6 | −0.109 | 0.225 | 0.295 | 0.993 | 1 |

**LPP-link at k=64 is the clear winner on the load-bearing metric:** link-Q
0.295 vs 0.240 baseline (+23 %) and link-flow 0.387 vs 0.311 (+24 %). Reprojecting
into the link-LPP subspace makes the embedding clusters respect the
human-curated link communities substantially better than raw GIST. There is a
**dimensionality sweet spot**: k=32 *underfits* badly (link-Q collapses to 0.142
— 32 link-LPP directions throw away too much of the embedding signal the
clusterer needs), and k=128 *over-includes* high-λ directions that re-import the
raw-GIST geometry and regress to ≈baseline (0.225). k=64 is the band where the
link-locality directions dominate without discarding usable signal.

The six singletons are the seed pages (Machine learning, NLP, Parsing,
Computational linguistics, Language model, Word embedding) in every row — they
carry the largest seeded PR mass and cluster alone regardless of the embedding
space, exactly as in `docs/wikipedia-curriculum.md`. This is expected and
confirms LPP touched only the clustering, not the graph-based selection.
Keyword distinctness is essentially flat across all variants (jacc ≈ 0.99,
#kw3+ ∈ {1,2,3}) — the topics were already keyword-distinct, so the win shows up
in *link structure*, not in keyword crowding.

### The qualitative win: infra-hub demotion + cleaner foundations

The improvement is legible in the topic tables (full tables in the machine
report). Under **raw GIST**, the top "AI/ML" cluster (topic 1, size 479) is
anchored by **bibliographic-infrastructure pages** mixed in with real content:
its top representative pages are *Semantic Scholar, ArXiv, PubMed Central*, Alan
Turing, Yoshua Bengio — the GIST embedding parks the citation-registry pages
right next to the AI researchers because their abstracts read similarly.

Under **LPP-link k=64**, that conflation is pulled apart:

- Topic 1 (size 529) becomes a clean **CS / algorithms / parsing** cluster:
  Programming language, Syntactic parsing, Algorithm, Parse tree, Regular
  expression. No infra hubs.
- The bibliographic-infrastructure pages are **demoted into lower-ranked, mixed
  clusters** (topic 11: DOI, Semantic Scholar; topic 12: ISSN, Bibcode, ArXiv) —
  they survive in the curriculum but no longer anchor a top topic, because the
  link graph does *not* place them in the conceptual core (their inbound links
  are hub-suppressed away).
- A clean **statistics** cluster (topic 10: Statistics, Regression analysis,
  Distributional semantics, Bias–variance tradeoff) and a **math/logic/semantics**
  cluster (topic 3: Formal grammar, Semantics, Formal language) separate out.

So link-LPP does at the *clustering* level a softer version of what
background-normalized PageRank does at the *selection* level: it pushes the
bibliographic infrastructure out of the conceptual foundations.

## Experiment 3 — link affinity vs kNN-of-embeddings affinity

Same nodes, same solver — only the affinity changes. The decisive number is the
**subspace overlap**: the mean canonical correlation (mean cosine of the
principal angles, via the SVD of `P_linkᵀ P_knn`) between the link-LPP and
kNN-LPP k=64 subspaces is **0.016** — essentially **orthogonal**. The
human-curated link graph and the embedding-similarity kNN graph induce almost
entirely **different** pedagogical subspaces. This is the crux result: the link
graph is not a noisy proxy for embedding similarity; it carries geometry the
GIST neighbors do not.

The kNN-LPP axes confirm it by capturing *surface embedding* structure, not link
communities — e.g. its axis 1 is ISO/airport country codes (ISO 3166-2:MV,
ICAO airport code, List of country codes) vs pop-music (Alexandra Burke, Josh
Groban): pages whose short abstracts are near-templated and so embed together,
with no pedagogical relationship. And on the load-bearing metric, **LPP-kNN k=64
does *not* sharpen the link clustering** (link-Q 0.225, link-flow 0.295 —
*below* raw GIST and well below LPP-link's 0.295/0.387). Reprojecting through the
embeddings' *own* neighborhood graph adds nothing over the embeddings; only the
*link* affinity does.

## Verdict

**Yes — LPP on the hub-suppressed Wikipedia link graph gives a useful
pedagogical subspace for Wikipedia's own curriculum, at k≈64.** The evidence:

1. **It measurably sharpens the clustering toward link structure.** LPP-link
   k=64 raises link-modularity Q from 0.240 → 0.295 (+23 %) and link-flow
   capture from 0.311 → 0.387 (+24 %) over raw GIST, with the same selection and
   clusterer. The clusters respect the human-curated link communities more.
2. **The qualitative effect is the right one.** It demotes the
   DOI/ISSN/ArXiv/Semantic-Scholar bibliographic-infrastructure pages out of the
   top AI/ML topic and surfaces clean CS-algorithms, statistics, and
   math/logic/semantics foundations — a softer, clustering-level echo of the
   background-norm fix.
3. **The signal is specifically from the links, not the embeddings.** LPP on the
   embedding kNN graph spans an almost orthogonal subspace (canonical corr 0.016)
   and does *not* improve link-clustering (it sits at/below baseline). So the
   gain is attributable to the human curation in the link graph, which is the
   whole hypothesis.

**Caveats / scope.**

- **Dimensionality matters and is dataset-specific.** k=32 underfits (link-Q
  collapses to 0.142) and k=128 regresses to baseline; k=64 is the sweet spot
  here. Any reuse should re-sweep k.
- **LPP captures community, not prerequisite *direction*.** The axes are
  topical/community contrasts, not depth axes — the symmetrized affinity throws
  away link direction. Prerequisite ordering remains the job of the separate
  frontier/voltage step in `docs/wikipedia-curriculum.md`, which this leaves
  untouched. The win here is *cleaner topic membership*, not *ordering*.
- **The improvement is real but moderate** (Q 0.24 → 0.30), and it does not
  dissolve the six seed singletons (a separate, graph-selection issue). This is a
  clustering-geometry refinement, not a wholesale curriculum rewrite.

The natural follow-on (left for the shared-space task that owns
`docs/wikipedia-openalex-shared-space.md`) is to fit the link-LPP projection on
Wikipedia and **apply it to OpenAlex embeddings** to get Wikipedia's pedagogical
geometry with OpenAlex's frontier coverage — the "best-of-both" in the backlog
entry.

## Reproduce

Reuses the cached subgraph; never touches the 43 GB dump. CPU-only.

```bash
conda run -p .conda python -m artifacts_scripts.wiki.wiki_pedagogical_lpp \
    --wiki-store cache/wiki/store/enwiki_full \
    --ranking-top-k 4000 --desired-n-topics 12 --use-abstracts
```

The machine report is written to
`artifacts/curriculum/wiki_pedagogical_lpp/report.md` (and `metrics.json`,
`P_link.npy`, per-variant `p2c_*.npy`); this curated doc is hand-authored from
those numbers and is **not** overwritten by re-runs.

**Performance note.** The link-LPP solves are cheap (each is one
`X_cᵀ L X_c` sparse-dense product, ~37 s, plus a 384×384 `eigh`). The slow step
is the kNN-of-embeddings affinity for Experiment 3 — an exact 477k×477k cosine
search. faiss's in-process OpenMP search-parallelism is unreliable on the test
box (it ran effectively single-threaded and ~10 min under CPU contention from a
co-scheduled job), so the code uses a high-recall approximate **IVFFlat** index
(√n cells, nprobe 64) instead; it still took ~4.6 min single-threaded here but is
robust and adequate (the binary affinity needs neighborhood community structure,
not ULP-exact neighbor ids). Total wall ~13 min under heavy contention; ~6–8 min
on an idle box.
