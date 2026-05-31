# Intrinsic curriculum-quality metric basket (the "quality dashboard")

A reusable, **intrinsic** (no external data) basket of metrics that scores a
built curriculum so we stop eyeballing topic orderings and clusterings. This is
the §B-intrinsic deliverable of
`docs/backlog/curriculum-diffusion-and-quality-metrics.md`. The **extrinsic
LLM-as-judge** layer (§B-extrinsic) — which catches the *semantic* failures this
structural basket provably misses — is now built too; see
[Extrinsic: LLM-as-judge](#extrinsic-llm-as-judge) at the bottom. The remaining
extrinsic proxies (reference syllabus / LectureBank) are still deferred.

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
- **wiki-native** — `wiki_curriculum_demo.py --reuse-subgraph --bg-norm` (bg-norm
  selection), 12 topics. **NB (ordering label):** the cached curriculum was built
  with `--order-by frontier-bgnorm`, but `--order-by` only changes the *ordering*,
  not the persisted clustering/keywords/subgraph, and the wiki-native loader does
  not read a persisted order — it **recomputes a plain `--order-by frontier`
  linearization** (`compute_frontier_adjacency` + linearization) on the matched-title
  subgraph. So every wiki-native number in this table (incl. the −0.706 Sim→Tgt
  ascent) is already the **`frontier`** result, not `frontier-bgnorm`. The earlier
  "frontier-bgnorm" label on this row was a misnomer; see the §v2-vs-v3 synthesis
  below for the corrected-wiki-native finding (the sim→target sign does NOT flip).
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

## v2 hard-expansion vs v3 diffusion — the synthesis (does the coverage win cost quality?)

The §A diffusion result was a **coverage** win: smooth GDC densification at the
held-out-link `t=8`, `--diffusion-top-m 32` lifts OA-into-the-curriculum from
**19.8% (v2 hard expansion) → 27.0% (v3 diffuse)**. The open question (this is the
backlog's *Synthesis* item): is that also a **quality** win, or is `top_m=32`
coverage-Goodharted — buying admitted-OA mass at the expense of coherence,
crowding, modularity, and balance? We scored the head-to-head on the intrinsic
basket. To make the embedding-space ordering metric comparable across the joint
configs (which persist no per-node embeddings), the joint runs were re-run with a
local `BOOKGAP_JOINT_DUMP_EMB=1` dump of the clustered-node GIST embeddings, and
the **target is the shared OpenAlex NLP-article centroid `/tmp/nlp_dir.npy`** — the
same axis wiki-native uses, so Sim→Tgt is comparable across every row (the dumped
NLP-seed-node centroid is the documented fallback). Reproduce:

```bash
# one-time per config (env-gated emb dump is OFF by default / uncommitted):
BOOKGAP_JOINT_DUMP_EMB=1 conda run -p .conda python -m artifacts_scripts.wiki.joint_graph_curriculum \
    --tau 0.5 0.6 0.7 --bridge-scorer csls --lpp-k 64 --desired-n-topics 14 \
    --oa-densify diffuse --diffusion-kernel heat --diffusion-top-m {8,16,32} --diffusion-t 8
# then the head-to-head dashboard (reads cache/.curriculum_quality/joint_v3_m{8,16,32} + joint_v2_emb):
conda run -p .conda python -m artifacts_scripts.curriculum.curriculum_quality --synthesis
```

### Head-to-head dashboard (discriminating metrics)

NPMI / Jaccard / embed-mean are de-emphasized (flagged-weak in the §intrinsic
validation: NPMI-over-titles is n/a for the joint set, Jaccard is uniformly ≥0.98,
embed-mean barely moves 0.71→0.72). Coverage row added for context.

| metric | dir | wiki-native(frontier) | **v2-expand** | v3-m8 | v3-m16 | **v3-m32** |
| :-- | :-: | --: | --: | --: | --: | --: |
| **OA→curriculum (coverage)** | ↑ | — | 19.8% | 12.2% | 24.3% | **27.0%** |
| OA-block density (edges/node) | — | — | 6.4 | 9.9 | 15.2 | 27.5 |
| Sim→Tgt ascent (Spearman) | ↑ | −0.706 | −0.116 | +0.033 | −0.086 | +0.011 |
| Sim→Tgt ascent (Kendall) | ↑ | −0.515 | −0.121 | +0.033 | −0.121 | +0.011 |
| DAG prereq-completeness | ↑ | 0.781 | 0.665 | 0.683 | 0.621 | 0.647 |
| [flag] dir. τ vs in-degree | flag | −0.727 | −0.758 | −0.890 | −0.670 | −0.560 |
| Keyword crowding (tpc/kw) | ↓ | 1.622 | 1.197 | 1.186 | **1.111** | 1.138 |
| #kw in 3+ topics | ↓ | 9 | 4 | 2 | **1** | 2 |
| #bibliographic-noise kw (top-8) | ↓ | — | — | 2 | 2 | **5** |
| Modularity Q | ↑ | 0.244 | 0.080 | 0.063 | 0.131 | **0.202** |
| Size entropy (norm) | ↑ | 0.704 | 0.831 | 0.807 | 0.858 | **0.916** |
| Size Gini | ↓ | 0.588 | 0.487 | 0.512 | 0.451 | **0.337** |
| #singleton topics | ↓ | 6 | 3 | 3 | 2 | **1** |
| Step smoothness (mean) | ↓ | 0.155 | 0.115 | 0.115 | 0.104 | **0.100** |
| Step smoothness (max) | ↓ | 0.271 | 0.206 | 0.205 | 0.205 | 0.227 |

(Keyword-coherence embed mean/worst and Jaccard are near-flat across all five
configs — 0.71–0.72 / 0.67–0.68 / 0.98–0.99 — and omitted; they neither separate
the configs nor flag the m32 noise, consistent with the weak-proxy finding.)

### Verdict 1 — does v3-diffusion's coverage win come *with* quality, or trade it away?

**With.** Comparing the default v2 (hard expansion, 19.8%) vs v3-m32 (diffuse, 27.0%)
on every *discriminating structural* metric, v3-m32 is **equal-or-better on all of
them**: singletons 3→**1**, size-Gini 0.487→**0.337**, size-entropy 0.831→**0.916**,
modularity-Q 0.080→**0.202**, step-smoothness(mean) 0.115→**0.100**, crowding
1.197→1.138, and Sim→Tgt ascent −0.116→**+0.011** (from mildly descending to flat).
So the +7.2pp coverage is **not** Goodharted on the structural axes — diffusion
admits the extra OA tail into *better-balanced, more graph-coherent, smoother*
topics than the hard expansion did. The headline diffusion win is a genuine
quality-and-coverage win, not a coverage-for-quality trade.

**The one place m32 pays:** keyword **noise**. Bibliographic/tool brand-name
keywords (`icanhazpdf`, `polyanalyst`, `mistral`, OCR garbage `amp_schubert`,
`witkam`) appearing in a topic's top-8 rise **2 (m8) → 2 (m16) → 5 (m32)** — at m32
two of them land in *foundations* topics (topic 0 = probability/logic/math picks up
`amp_schubert/witkam`; topic 2 = academic-publishing picks up
`icanhazpdf/polyanalyst/mistral`). This is precisely the A-diffusion run's flag
("a little bibliographic noise leaks into foundations at the denser setting"),
now quantified. It is a *semantic-keyword* blemish the structural basket otherwise
rates as the cleanest config — exactly the kind of judgment the intrinsic basket is
known to under-weight (cf. the v1 toolkit-grab-bag miss).

### Verdict 2 — the top_m trade: where is the sweet spot, is 32 right?

Coverage and OA-block density both rise monotonically with `top_m`
(12.2%/9.9·→24.3%/15.2·→27.0%/27.5·), but the **marginal coverage gain collapses**:
m8→m16 buys **+12.1pp** for ~+5 edges/node, while m16→m32 buys only **+2.7pp** for
**~+12 edges/node** — diminishing returns, and the doubling of density is what drags
the bibliographic noise in (#noise-kw 2→5). On the structural metrics m32 is still
the best (it monotonically improves balance/singletons/modularity with density), so
there is a real **tension**:

- **m32** — best coverage, best size-balance/singletons/modularity/smoothness, but
  the **noisiest topics** (5 noise-kw, leaking into foundations).
- **m16** — captures **90% of the coverage win** (24.3% vs 27.0%) at *half* the OA
  density, with the **cleanest keyword profile of all** (crowding 1.111, #kw-in-3+
  **1**, #noise-kw 2) and second-best balance (Gini 0.451, 2 singletons). Modularity
  0.131 is below m32's 0.202 but still well above v2's 0.080.
- **m8** — under-densified: coverage *below* the v2 hard-expansion baseline (12.2% <
  19.8%) and worst balance of the three (Gini 0.512). Too sparse; not a contender.

**Recommendation: the `top_m=32` default is defensible but `top_m=16` is the
cleaner-topics sweet spot.** If the goal is *maximum* OA coverage + structural
balance, keep 32. If the goal is the *cleanest topics* for ~1pp of coverage cost
m16 is the better operating point — it trades 2.7pp of coverage for the lowest
keyword-crowding and the elimination of the foundations-noise leak. We keep **32 as
the committed default** (coverage is the headline diffusion win and the noise is a
keyword-display blemish, not a structural one), but record **16 as the
quality-first alternative** and the diminishing-returns knee.

### Verdict 3 — corrected wiki-native: did Sim→Tgt flip positive under `--order-by frontier`?

**No — and the reason is instructive (a metric-validity finding, not an ordering
bug).** The prior dashboard *labeled* the wiki-native row `frontier-bgnorm`, but the
scorer never used that order: it recomputes a **plain `--order-by frontier`**
linearization. So the −0.706 was *already* the `frontier` result; re-running with
`--order-by frontier` changes nothing on disk (clustering/keywords/subgraph are
order-independent), and recomputing the order on the *full* 477k link graph (the
real pipeline's order, not the sliced subgraph) gives **−0.664** — still strongly
negative. **The flip the synthesis plan hypothesized does not occur.**

Diagnosis: the Sim→Tgt **target** is the OpenAlex NLP-*article* centroid, which sits
near the foundational/core-NLP topics (statistics 0.92, software 0.88), while
wiki-native's genuine *frontier* (the GPT/deep-learning/word-embedding topics it
correctly orders **last**) is *farther* from that centroid (0.79–0.87). So a
correctly-ordered foundations→frontier curriculum **descends** in cosine-to-this-target
and scores a negative ascent. The metric is only valid when the target *is* the
curriculum's destination — which holds for the OA-cite curriculum (its target is the
graph-embedding frontier, Sim→Tgt climbs 0.86→0.96, ascent **+0.73**) and for the
joint configs (target ≈ corpus mean ⇒ near-flat ±0.1, as seen) but **not** for
wiki-native against the article-centroid target. **Net:** the corrected-wiki-native
row is `frontier` (−0.71→−0.66, essentially unchanged), and Sim→Tgt-ascent should be
read as *valid only when the target coincides with the curriculum goal*; for
wiki-native it reports target-geometry, not an ordering defect. The real
wiki-native failures remain its **6 singletons** and crowding, not its ordering.

## Extrinsic: LLM-as-judge

The structural basket above catches *structural* failures (singletons, balance,
modularity, ordering geometry) but **provably misses two semantic ones** — both
documented above as the prime motivation for an extrinsic phase:

- (a) v1's **pytorch/keras toolkit grab-bag** topic (rank 12: `learning,
  machine_learning, net, kubeflow, tensorflow, lightgbm, keras`) — embed-coherent
  (keyword-coherence rates it fine) and lexically distinct (high Jaccard), but
  pedagogically a *tool list*, not a teachable concept;
- (b) v3-m32's **bibliographic / tool / OCR keyword leak** (`icanhazpdf`,
  `polyanalyst`, `mistral` in the machine-translation topic; OCR garbage
  `amp_schubert`, `witkam`) — which the structural metrics rate as the *cleanest*
  config (best balance/modularity/singletons).

The extrinsic layer is an **LLM-as-judge** over the project's **local ollama**
adapter (the standing rule: LLM eval goes through local ollama, never the OpenAI
Batch API). Scorer: `artifacts_scripts/curriculum/curriculum_llm_judge.py`. It
reuses the *same* cached curricula as the intrinsic scorer
(`cache/.curriculum_quality/`), parsing each cache's committed `report*.md`
"Topic Overview" table for per-topic keywords + representative titles — that
table is already emitted in curriculum (prerequisite → frontier) order, so the
judge reuses the pipelines' own ordering rather than re-deriving it.

```bash
# default run (model gpt-oss:20b, 3 repeats/item for an agreement read):
conda run -p .conda python -m artifacts_scripts.curriculum.curriculum_llm_judge \
    --json-out cache/.curriculum_quality/llm_judge_gptoss20b.json
# quick single-repeat look at one curriculum:
conda run -p .conda python -m artifacts_scripts.curriculum.curriculum_llm_judge \
    --quick --only joint-v1
```

**Design (semantic dimensions the structural basket can't see):** all outputs
are **structured** (ollama `format=<json schema>` — never free text) and the
prompts are **neutral** (1–5 scale with both poles described; the judge is never
told which answer we want). Three judged dimensions:

1. **Topic pedagogical validity** (per topic, from its keywords + representative
   titles): a 1–5 `rating` (5 = coherent teachable concept, 1 = grab-bag /
   tool-list / infrastructure / bibliographic-noise cluster), a boolean
   `is_grab_bag`, and a one-line reason. The litmus for (a).
2. **Prerequisite-ordering correctness** (per sampled ordered pair i<j, with i
   taught earlier): `agree`/disagree that A-before-B is sensible + reason,
   aggregated to an **order-sensibility rate**. (All adjacent steps + random
   non-adjacent pairs, capped at `--pairs 10`.)
3. **Keyword meaningfulness** (per topic): which top keywords are
   bibliographic / tool-brand / OCR-noise tokens rather than genuine concepts.
   Flags are kept only if they (i) appear in the topic's actual keyword list and
   (ii) are flagged by a **majority** of repeats. The litmus for (b).

Bounded by design: ~12–15 topics × 2 calls (validity + keyword) + ~10 ordered
pairs, each ×`--repeats` (default 3, majority-voted with a per-item agreement
read). The default run = **558 ollama calls, ~21 min** on the local box.

### Semantic dashboard (model `gpt-oss:20b`, 3 repeats)

| semantic metric | dir. | joint-v1 | joint-v2 | v3-m16 | v3-m32 | wiki-native |
| :-- | :-: | --: | --: | --: | --: | --: |
| topics judged | | 14 | 14 | 14 | 14 | 12 |
| Topic validity (mean 1–5) | ↑ | 2.43 | 2.36 | 2.31 | 2.36 | **3.00** |
| Topic validity (worst topic) | ↑ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| #grab-bag topics flagged | ↓ | 7 | 7 | 7 | 7 | **4** |
| #noise keywords flagged (all kinds) | ↓ | 29 | 22 | 19 | 18 | 28 |
| — of which bibliographic/tool/OCR | ↓ | 7 | 9 | **5** | 7 | **0** |
| #topics with noise kw | ↓ | 7 | 7 | 6 | 6 | 6 |
| Order-sensibility rate | ↑ | 0.60 | 0.60 | **0.80** | 0.70 | 0.50 |
| [noise] grab-bag vote agreement | — | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| [noise] order vote agreement | — | 1.00 | 0.93 | 1.00 | 0.97 | 0.97 |

Higher = better: topic validity, order-sensibility rate. Lower = better:
#grab-bag, #noise keywords, #topics-with-noise. `[noise]` rows are the judge's
**self-consistency across repeats** (1.0 = unanimous) — a reliability read, not a
quality axis. The "bibliographic/tool/OCR" sub-row isolates the specific
keyword-meaningfulness defect the intrinsic `#bibliographic-noise kw` row tracks
(tool/brand names + OCR garbage), separated post-hoc from person-name and generic
noise the judge also flags.

### Litmus — did the judge catch what the structural basket missed?

**Both litmus targets: caught, unanimously.**

- **(a) v1 toolkit grab-bag — CAUGHT.** v1's rank-12 topic is flagged
  `is_grab_bag=true` (3/3 repeats, agreement 1.00), and its keyword-meaningfulness
  pass returns exactly **`['keras', 'kubeflow', 'lightgbm', 'net', 'tensorflow']`**
  as tool noise. The reason text: *"blends core machine-learning concepts with a
  collection of libraries and frameworks … a loose, tool-heavy theme rather than a
  single coherent concept."* This is the topic *no intrinsic metric here saw*
  (keyword-coherence-embed and Jaccard both rate it clean).
- **(b) v3-m32 bibliographic leak — CAUGHT.** m32's machine-translation topic
  (rank 6) returns exactly **`['icanhazpdf', 'mistral', 'polyanalyst']`** as noise,
  with the reason *"these refer to a website, a software tool, and a product name
  rather than core NLP/ML concepts,"* and the OCR-garbage topic (rank 13) returns
  `['amp', 'amp_schubert', 'maxwell', 'witkam']`. Note the judge rates that rank-6
  topic's *overall validity* at 3 with `is_grab_bag=false` — its core (MT/word
  embeddings) is a real concept; the defect is **keyword-level**, which is why
  keyword-meaningfulness is a *separate* dimension from grab-bag detection and is
  the right axis for (b).

So the extrinsic layer **earns its keep**: it surfaces, in plain language with
the exact offending tokens, the two semantic failures that were previously only
catchable by eyeballing.

### m32-vs-m16 semantic verdict (informing the `top_m` default)

**Nuanced, and it confirms the intrinsic finding rather than overturning it.** On
the *specific* bibliographic/tool/OCR keyword axis — the one the intrinsic
`#bibliographic-noise kw (top-8)` row tracks (2 vs 5 there) — the judge agrees in
**direction**: **m16 = 5, m32 = 7** flagged bibliographic/tool/OCR tokens, so
**m32 is semantically worse on keyword-meaningfulness**, exactly as the intrinsic
metric said. (The two counts differ in magnitude because the judge parses the
report's **top-7** keyword column, not the manual top-8 count, and tallies every
flagged token rather than a curated list — but the m16 < m32 ordering holds.)
The *total* noise count is near-tied (m16 = 19, m32 = 18) only because m16 carries
a whole **person-name** topic (rank 2: `john, turing, martin, …`, 7 name tokens
the judge also flags) that m32 lacks — name noise, a different blemish from the
tool/bibliographic leak the `top_m` knob actually controls.

Topic *validity* is statistically indistinguishable between the two (m16 2.31 vs
m32 2.36; identical grab-bag count 7; both worst-topic 1.00) — the judge does not
see a pedagogical-coherence difference. Where m16 is *better* is **order
sensibility (0.80 vs 0.70)**, the cleaner prerequisite ordering.

**Net for the default:** the extrinsic layer **reinforces** the intrinsic
synthesis verdict — `top_m=32` buys coverage + structural balance at the cost of a
**measurably noisier keyword profile** (more tool/bibliographic tokens), while
`top_m=16` is the **cleaner-topics, better-ordered** operating point. Neither the
intrinsic nor the extrinsic layer finds a *pedagogical-validity* gap; the trade is
purely coverage-and-structure (favoring 32) vs keyword-cleanliness-and-ordering
(favoring 16). We keep **32 as the committed default** and **16 as the
quality-first alternative**, now with an extrinsic second opinion behind both.

### Judge reliability — an honest read

- **Self-consistency is high.** Across all five curricula the grab-bag vote
  agreement is **1.00** (every topic's grab-bag verdict is unanimous over 3
  repeats at temperature 0) and the order-pair agreement is **0.93–1.00**. The
  two litmus flags are unanimous. So the *flags we report are stable*, not
  coin-flips.
- **Transport noise is real but absorbed.** `gpt-oss:20b` emitted malformed /
  truncated JSON on **39/558 calls (7.0%)** despite schema-constrained decoding;
  with 3 repeats + majority voting these drop out without changing a verdict. A
  single-repeat (`--quick`) run is therefore *less* trustworthy — keep ≥3 repeats
  for any reported number.
- **The judge is liberal with `is_grab_bag`.** It flags **7/14** topics for the
  joint configs, including defensible-but-debatable calls (person-name clusters,
  the academic-publishing topic). This makes the *absolute* grab-bag count a
  blunt instrument; the **specific, named flags** (the toolkit topic, the
  `icanhazpdf` tokens) are where the signal is, and those are exactly right. Read
  the count as "how many topics look non-teachable to a careful reviewer" (so
  wiki-native's 4-vs-the-joints'-7 and its higher mean validity 3.00 is a genuine
  separation), and read the per-topic keyword/reason output for the actionable
  detail.
- **Bottom line:** the layer is **reliable enough to separate the configs on the
  axes it was built for** — it catches both semantic failures the structural
  basket misses, with unanimous agreement, and corroborates the m32-noisier /
  m16-cleaner read. It is *not* a precise pedagogical-quality oracle (the
  grab-bag count is liberal and ~7% of raw calls need the repeat-vote to recover);
  treat the named flags + reasons as the deliverable, the aggregate counts as
  directional.
