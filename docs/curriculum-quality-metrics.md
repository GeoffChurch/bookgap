# Intrinsic curriculum-quality metric basket (the "quality dashboard")

A reusable, **intrinsic** (no external data) basket of metrics that scores a
built curriculum so we stop eyeballing topic orderings and clusterings. This is
the §B-intrinsic deliverable of
`docs/backlog/curriculum-diffusion-and-quality-metrics.md`. Extrinsic proxies
(reference syllabus / LectureBank / LLM-as-judge) are deferred.

- **Scorer:** `artifacts_scripts/curriculum/curriculum_quality.py`. The core is a
  *pure* function `score_curriculum(CurriculumInputs) -> dict[str, float]` (arrays
  in, metrics out, no disk/Cloud access). Per-curriculum loaders + a dashboard CLI
  live in the same module.
- **Reuse:** clustering metrics (`modularity_q`, `cluster_size_gini`,
  `keyword_distinctness`, `avg_pairwise_jaccard_distance`) come straight from
  `compare_clusterings.py`; the inter-topic dependency matrix (for DAG-prereq and
  in-degree directionality) is built with the pipelines' own
  `pagerank.compute_frontier_adjacency` + `build_curriculum._compute_linearization_order_from_adjacency`;
  NPMI tokenization reuses `keyword_extraction.tokenize`/`filter_stopwords`.

```bash
# Fast path (graph + keyword + embedding-keyword-coherence metrics):
conda run -p .conda python -m artifacts_scripts.curriculum.curriculum_quality

# Add OA NPMI doc tokens (loads the Cloud for titles; slow):
conda run -p .conda python -m artifacts_scripts.curriculum.curriculum_quality --with-embeddings
```

The CLI reads cached curriculum artifacts from
`cache/.curriculum_quality/{wiki_native,joint_v1,joint_v2}/` plus the OA-citation
run at `artifacts/curriculum/gap_curriculum_T11273/`. The cache was populated by
running the **committed** scripts once up front (the joint script overwrites a
single `ART_DIR`, so v1 and v2 were each run once and copied aside; the wiki-native
script's `wiki_nlp_cirrus/` outputs were copied too). We do **not** re-invoke the
separately-owned `joint_graph_curriculum.py` from the scorer.

## Metric definitions and directions

| Metric | Axis | Direction | What it measures |
| :-- | :-- | :-- | :-- |
| `sim_to_target_spearman` / `_kendall` | ordering | **higher** (→+1) | rank-correlation of curriculum position vs each topic's cosine-to-target-centroid. A curriculum should *ascend* toward the goal, so later topics should be more target-similar. |
| `dag_prereq_completeness` | ordering | **higher** (→1) | on the directed topic graph (`adj[i,j]` = topic i depends on j), the dependency-mass-weighted fraction of each topic's prerequisite in-neighbours that are ordered *earlier*. No knobs. |
| `directionality_indegree_kendall` | ordering | **flag** | Kendall-τ of position vs topic in-degree. Foundations (high in-degree) should be early ⇒ a *positive* τ flags frontier-first. Report, don't optimize. |
| `directionality_pubyear_kendall` | ordering | **flag** | Kendall-τ of position vs median pub-year. *Negative* τ = later topics are older = frontier-first failure. Needs pub-years (n/a for wiki / joint here). |
| `topic_coherence_npmi` | clustering | higher | mean intra-topic NPMI of top-K keyword pairs, using the curriculum's own member documents as the reference corpus (Mimno top-K NPMI). **Weak as implemented** — see below. |
| `keyword_coherence_embed` (mean) / `_min` (worst) | clustering | **higher** | mean / minimum-over-topics intra-topic pairwise cosine of top-K keyword *embeddings* (GIST-small). The `_min` is meant to surface a single grab-bag topic the mean washes out. |
| `cross_topic_distinctness_jaccard` | clustering | higher (→1) | avg pairwise Jaccard *distance* of top-K keyword sets across topics. |
| `avg_topics_per_keyword` | clustering | **lower** (→1) | keyword crowding: 1.0 = perfect distinctness, >1 = top-K keyword overlap across topics. |
| `n_keywords_in_3plus_topics` | clustering | **lower** | count of keywords appearing in ≥3 topics' top-K. |
| `modularity_q` | clustering | higher | Newman modularity of the topic partition on the (symmetrised) graph. Never optimize alone (max-Q fragments). |
| `size_entropy_norm` | clustering | **higher** (→1) | Shannon entropy of the size distribution, normalized to [0,1] (1 = uniform sizes). |
| `size_gini` | clustering | **lower** (→0) | Gini of cluster sizes (~1 = one mega-cluster). |
| `n_singletons` | clustering | **lower** | count of size-1 topics (the seed-as-singleton / degenerate-OA failure). |
| `step_smoothness_mean` / `_max` | path | **lower** | mean / max consecutive-topic centroid distance (1−cosine) along the order; a big jump flags a missing prerequisite. Needs topic centroids. |

Metrics whose inputs are unavailable for a curriculum return `NaN` (shown `n/a`).
The joint v1/v2 runs persist no per-node embeddings or titles for their clustered
set, so their embedding-space metrics (sim→target, step-smoothness) and NPMI docs
are `n/a`; their graph + keyword + embedding-keyword-coherence metrics carry the
comparison. Wikipedia pages have no publication year, so the pub-year flag is
`n/a` for the wiki-side curricula.

## Dashboard (curriculum × metric)

Numbers from the default run plus `--with-embeddings` for the OA NPMI cell.

| metric | wiki-native | joint-v1 | joint-v2 | OA-cite-T11273 |
| :-- | --: | --: | --: | --: |
| Sim→Tgt ascent (Spearman) ↑ | **−0.706** | n/a | n/a | **+0.727** |
| Sim→Tgt ascent (Kendall) ↑ | −0.515 | n/a | n/a | +0.498 |
| DAG prereq-completeness ↑ | 0.781 | 0.665 | 0.665 | 0.679 |
| dir. τ vs in-degree (flag) | −0.727 | −0.736 | −0.758 | −0.562 |
| dir. τ vs pub-year (flag) | n/a | n/a | n/a | n/a |
| Topic coherence NPMI ↑ | −0.574 | n/a | n/a | −0.353 |
| Keyword coherence (embed, mean) ↑ | 0.696 | 0.709 | 0.715 | 0.733 |
| Keyword coherence (embed, worst) ↑ | 0.672 | 0.666 | 0.673 | 0.705 |
| Cross-topic distinctness (Jacc) ↑ | 0.837 | 0.986 | 0.982 | 0.914 |
| Keyword crowding (tpc/kw) ↓ | 1.622 | 1.157 | 1.197 | 1.807 |
| #kw in 3+ topics ↓ | 9 | 2 | 4 | 16 |
| Modularity Q ↑ | 0.244 | 0.092 | 0.080 | 0.260 |
| Size entropy (norm) ↑ | 0.704 | 0.886 | 0.831 | 0.964 |
| Size Gini ↓ | 0.588 | 0.379 | **0.487** | 0.245 |
| #singleton topics ↓ | **6** | 3 | 3 | **0** |
| Step smoothness (mean) ↓ | 0.155 | n/a | n/a | n/a |
| Step smoothness (max) ↓ | 0.271 | n/a | n/a | n/a |

The curricula scored:
- **wiki-native** — `wiki_curriculum_demo.py --reuse-subgraph --bg-norm --order-by
  frontier-bgnorm` (bg-norm selection + frontier-bgnorm ordering), 12 topics.
- **joint-v1** — `joint_graph_curriculum.py --no-oa-expand --no-per-corpus-select`
  (the OA-mass-starved baseline; 247/1941 OA reach a topic = 12.7%), 14 topics.
- **joint-v2** — `joint_graph_curriculum.py` defaults (OA-expand +
  per-corpus-select; 644/3260 OA reach a topic = 19.8%), 14 topics.
- **OA-cite-T11273** — `artifacts/curriculum/gap_curriculum_T11273/`, the
  OpenAlex-citation gap curriculum, 15 topics.

## Which metrics discriminate — and which are weak

**Strong discriminators (trust these):**

- **`n_singletons` + `size_gini` + `size_entropy_norm`** cleanly separate the
  curricula. OA-cite is the cleanest (0 singletons, Gini 0.25, entropy 0.96);
  **wiki-native is the worst** (6 singletons — the six seed pages each form their
  own size-1 topic — Gini 0.59); both joint runs carry 3 singleton/degenerate-OA
  topics. This catches exactly the giant-cluster + singleton pathologies the
  design called out.
- **`sim_to_target_spearman/kendall`** is decisive where computable: OA-cite
  **ascends** toward its target (+0.73 / +0.50 — matches the precomputed Sim→Target
  column climbing 0.86→0.96 in its `summary.md`), whereas **wiki-native descends**
  (−0.71 / −0.52). The wiki-native reconstruction puts its GPT/deep-learning
  frontier topic at *rank 1* and foundations later — a **frontier-first ordering
  failure**, surfaced by a strongly negative ascent. (n/a for the joint runs:
  no per-node embeddings persisted.)
- **`modularity_q`** separates graph-coherent partitions (OA-cite 0.26, wiki-native
  0.24) from the bridge-blurred joint graphs (0.08–0.09) — expected, since the
  cross-corpus bt_cos bridge edges deliberately smear community boundaries.
- **`keyword crowding` / `#kw in 3+ topics`** discriminate sensibly: the joint runs
  are least crowded (tpc/kw ≈ 1.16–1.20) and **OA-cite is the most crowded**
  (1.81, 16 keywords in ≥3 topics) — its topics share generic ML vocabulary
  (learning / network / graph / embedding recur across topics).
- **`dag_prereq_completeness`** modestly favors wiki-native (0.78) over the joint
  runs (0.665) and OA-cite (0.68) — wiki's link backbone orders prerequisites
  before dependents slightly better. A useful soft signal.

**Expected signals — confirmed and refuted:**

- *v2 ≥ v1 on coverage* — **confirmed externally** (644 vs 247 OA into the
  curriculum, the headline v2 win) but **NOT a balance win**: on cluster-size
  balance the basket shows **v1 slightly better** (Gini 0.38 vs 0.49; entropy 0.89
  vs 0.83), and singleton count is tied at 3. **Finding:** v2's gain is *coverage*
  (more OA papers admitted), not *balance* — admitting the extra OA tail makes the
  size distribution marginally less even, not more. The design's "v2 ≥ v1 on …
  balance" expectation does not hold; coverage is where v2 wins.
- *the v1 pytorch/keras toolkit topic should show low coherence/distinctness* —
  **REFUTED by the intrinsic basket.** The v1 toolkit topic (keywords
  `learning, machine_learning, net, kubeflow, tensorflow, lightgbm, keras,
  yooreeka, xgboost, perceptron`) scores embed-coherence **0.712 — above the v1
  mean** and is lexically distinct from every other topic (so Jaccard distinctness
  doesn't flag it either). The reason: GIST embeds the tool *brand names* as
  mutual neighbours, so an embedding-cosine metric reads the topic as coherent.
  Its actual defect — a list of library brand-names rather than a *concept* — is a
  human pedagogical judgment that **no intrinsic embedding/lexical metric in this
  basket captures.** This is the clearest "weak proxy" result: flag it for the
  extrinsic (LLM-judge / syllabus) phase.
- *wiki-native clean on coherence* — **partly.** Its keyword embed-coherence (0.70
  mean / 0.67 worst) is in line with the others, but its *ordering* (frontier-first,
  Sim→Tgt −0.71) and *size balance* (6 singletons) are its real failures — coherence
  was never wiki-native's problem; ordering and singletons are.
- *frontier-first ordering shows negative directionality-τ* — the **pub-year**
  check is n/a here (no years for wiki/joint), and the **in-degree** τ is negative
  for all four (high-in-degree/foundational topics land early *by link/citation
  degree*). The frontier-first failure wiki-native does have is caught instead by
  the **Sim→Target ascent** going negative. **Finding:** the in-degree
  directionality flag and the sim-to-target ascent measure *different* notions of
  "foundational" (graph hub-ness vs semantic distance-to-goal) and can disagree;
  the sim-to-target ascent was the more sensitive frontier-first detector here.

**Weak proxies (flagged, do not over-trust):**

- **`topic_coherence_npmi`** — computed over per-topic *titles* (the only member
  text on disk), which are too short to give reliable co-occurrence: both
  reasonable curricula score *negative* (wiki −0.57, OA −0.35) and it is n/a for
  the joint runs (no member texts cached). It does not separate good from bad here.
  A proper NPMI needs a denser reference corpus (page bodies / abstracts) — left
  for the extrinsic phase. **Treat as not-yet-useful.**
- **`keyword_coherence_embed` (mean and worst)** — the *min* successfully flags
  garbage/OCR singleton topics (the v1 `amp/amp_schubert/maxwell` topic is its
  worst), but it does **not** flag the toolkit grab-bag (above), and the mean
  barely varies across curricula (0.70–0.73). Useful as a *singleton/garbage*
  detector, not as a general coherence ranker.
- **`cross_topic_distinctness_jaccard`** — uniformly high (0.84–0.99) and does not
  isolate the single-bad-topic failures (a grab-bag topic that is *lexically*
  unique still scores as distinct). The crowding metrics (`avg_topics_per_keyword`,
  `#kw in 3+ topics`) carry the distinctness signal better.

## Takeaways

The basket reliably catches **structural failures** — singletons, size imbalance,
mega-clusters (`n_singletons` / `size_gini` / `size_entropy_norm`), graph
fragmentation (`modularity_q`), keyword crowding, and **ordering** failures where
embeddings are available (`sim_to_target` ascent, `dag_prereq_completeness`). It is
**blind to topic-level *semantic* coherence judgments** — the
toolkit-brand-name-grab-bag is the canonical miss — and its lexical-coherence
proxy (NPMI over titles) is too sparse to be useful. Those gaps are the motivation
for the deferred extrinsic anchor (syllabus / LectureBank / LLM-judge): use the
intrinsic basket to gate structure, and the extrinsic phase to judge the semantics
the intrinsic metrics provably cannot see.
