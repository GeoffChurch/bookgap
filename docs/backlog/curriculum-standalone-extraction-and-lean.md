# Standalone curriculum library + mathlib4/Lean application

**Living doc — take it up cold; update as worked through.** Migration intent: once
extraction (Phase 1) is done, **move this doc to `~/src/curriculum/`** (the new standalone
library), then **split the Lean-specific material (Phases 2–4) into a backlog in
`~/src/curriculum_lean/`**. While extraction hasn't happened, it lives here in bookgap.

Self-contained: a fresh session should be able to act on this without the originating
conversation. Companion (engine history + validated graph-generic features):
`docs/backlog/curriculum-generalization-pedagogical-subspace.md`.

## Goal

Extract bookgap's curriculum engine into a **standalone, input-agnostic library** (`~/src/curriculum`)
that turns a `(graph, [optional embeddings], seeds, node-text)` input into an ordered,
clustered, keyworded curriculum. Then test a concrete hypothesis on a new domain:

> Does an agent fix/complete broken **mathlib4** libraries more quickly / efficiently / reliably
> given a **curriculum built from the proof-dependency graph**, vs a simple-retrieval baseline,
> vs no extra context? (And later: the same for bug-fixing in large **Rust** codebases, from the
> code-dependency graph.)

## Phase 0 — what the engine is (so extraction is concrete)

The engine only ever needs a **`(sparse graph, node embeddings, seed nodes, per-node text)`**
quadruple, then runs: **seeded/background-normalized PageRank** (relevance) → **top-k selection**
(per-corpus-normalizable) → **clustering** (topics) → **frontier-adjacency directional ordering**
(prerequisite → frontier) → **keyword extraction** → **summary + quality metrics**. Validated this
session over **four different graphs** (OpenAlex citations, Wikipedia links, a joint wiki+OA graph,
GDC-diffusion-rewired variants) — i.e. the source-agnosticism is already demonstrated, not
hypothetical.

**Source modules in bookgap** (`artifacts_scripts/curriculum/` unless noted):
- `pagerank.py` — seeded PageRank, `two_direction_pagerank`, `compute_frontier_adjacency`
  (inter-topic dependency via absorbing PageRank).
- `build_curriculum.py` — ranking → top-k → cluster → `_compute_linearization_order_from_adjacency`
  (in−out-degree linearization) and `_compute_linearization_order_from_potentials` (voltage/CircuitRank).
- `circuitrank.py` — voltage/current (source→target) ordering.
- `similarity_graph.py` — kNN-of-embeddings affinity (`build_similarity_adjacency_from_*`).
- `compare_clusterings.py` — modularity / Gini / NPMI / pairwise-Jaccard.
- `keyword_extraction.py` — tokenization + log-odds / Fightin'-Words keywords.
- `curriculum_quality.py` — the **intrinsic quality basket** (Sim→Target ascent, DAG
  prereq-completeness, coherence, distinctness, modularity, size-balance, step-smoothness).
- `curriculum_llm_judge.py` — the **extrinsic LLM-as-judge** (ollama; grab-bag / prereq-order /
  keyword-noise; model-keyed, repeat-voted, raw-persist-on-failure).
- `seed_from_gap.py` — bookgap seed driver (gap → seeds); will be *replaced* by domain seed adapters.
- Wikipedia-side (graph-generic pieces): `bookgap/wikipedia/link_graph.py`,
  `artifacts_scripts/wiki/wiki_pedagogical_lpp.py` (**link-LPP**), and
  `artifacts_scripts/wiki/joint_graph_curriculum.py` (bg-norm selection, per-corpus rebalancing,
  **GDC diffusion densification** with held-out-link T-selection, joint-graph assembly).

**bookgap coupling to strip:** `Cloud`/`SubCloud` + OpenAlex scopes/enums (data adapter), GIST
sentence-transformers (the embedder), `util.py` singletons (`LOGGER`/`TIMER`/`MEMORY`/`ARTIFACTS_DIR`,
the joblib cache), gap-detection (the seed producer). All of this is a *thin shell* around the
graph+embedding core.

## Phase 1 — extract `~/src/curriculum` (input-agnostic)

1. Define a **`GraphSource` protocol**: `→ adjacency (scipy CSR), node_embeddings (optional),
   id↔index map, node_text, seed_nodes`. Plus a `CurriculumConfig` (ranking/ordering/clustering/
   bg-norm/diffusion knobs).
2. **Lift the algorithm modules** above; **strip the `util.py` singletons** (inject a logger/cache
   or use stdlib); **make the embedder pluggable** (accept precomputed embeddings, OR a callable, OR
   **none** → derive from the graph; see Phase 0-bis).
3. **Parity gate:** reproduce a known bookgap curriculum (e.g. the wiki-native or an OpenAlex one)
   through the new lib's `GraphSource` adapter — byte-for-byte topics/order — to prove the lift is
   behavior-preserving. (Same discipline as the SP1 gap-parity gate.)
4. On completion: **move this doc + the lib to `~/src/curriculum/`**.

**Minimal-core first.** The Lean-extractive use (Phase 4) needs only **rank (PageRank) + order
(topological / frontier-adjacency) + select (top-N)** — *no* clustering, keywords, embeddings, or LPP.
So extract that minimal core first (it's tiny and unblocks the whole Lean experiment), and add
clustering / keyword extraction / link-LPP / GDC diffusion / the quality basket as **optional layers**
needed only by embedding-rich domains. Don't gate the Lean work on lifting the full engine.

### Phase 0-bis — pure-graph mode (no semantic embedder): meaning from dependency alone
The embedder is **optional**. For domains where the graph *is* the data (Lean — see below), use
**trivial/identity/one-hot node features** and let the dependency structure carry all meaning
(pure distributional semantics). This is already well-founded in the code we built:

- **Identity features + link-LPP = Laplacian Eigenmaps on the graph.** With `X = I`, the link-LPP
  generalized eigenproblem `X L Xᵀ a = λ X D Xᵀ a` collapses to `L a = λ D a` — the normalized-Laplacian
  spectral embedding. So "node geometry from the graph alone" is exactly link-LPP with identity input.
  *Efficiency:* don't form the dense `X L Xᵀ`; compute the bottom-k eigenvectors of the (sparse,
  hub-suppressed) normalized Laplacian directly via `scipy.sparse.linalg.eigsh`. That's the spectral
  embedding; cluster in it.
- **Or skip embeddings entirely** and cluster the graph directly (Louvain/Leiden/spectral community
  detection — the engine already has the modularity machinery). Ordering stays **frontier-adjacency**
  (already pure-graph); keywords come from **node text** (declaration names/statements).

So `CurriculumConfig.embeddings ∈ {provided, callable, spectral-from-graph, none→community-detect}`.
Lean's default = spectral-from-graph (identity features). A semantic-embedding ablation (embed
theorem statements) is a *later* comparison, not a prerequisite.

**LPP's role in extractive mode (the Phase-4 question): repurposed for *selection geometry*, not
clustering, and optional.** With no clustering step, the spectral embedding (= identity-feature
link-LPP) instead informs *which* lemmas to select: seeded-PageRank says which are **load-bearing**;
the spectral geometry says how they're **arranged**, so you can pick a subset that **spans** T's
prerequisite manifold rather than a redundant pile from one sub-area. But the **minimal core is just
seeded-PageRank-select + topological/frontier-order** — add the spectral-diversity term (or the
equivalent community-detect-then-top-per-community) only if pure top-PageRank selection proves
redundant. Empirical; start without it.

## Phase 2 — mathlib4 / Lean `GraphSource` (the first new domain; easiest + best fit)

Why Lean first (over Rust): (1) the **proof-dependency graph is first-class and pre-extracted**
(Lean environment; **LeanDojo** ships traced mathlib with premise/dependency edges) — no
static-analysis tooling to build; (2) **verification is automatic, binary, trustworthy** (Lean
type-checks proofs) — the clean success signal; (3) **synthetic broken tasks are trivial** to
generate; (4) an **agent/benchmark ecosystem exists** (LeanDojo + ReProver, miniF2F, ProofNet) to
plug into; (5) the proof-dependency graph is an **exact prerequisite DAG** (decl→premises is logical,
not the noisy "built-on" of citations) — so "foundations→frontier" is *literally* correct.

`GraphSource` for Lean:
- **graph** = declaration→premise dependency edges (from LeanDojo traces / the Lean environment).
- **embeddings** = identity → spectral-from-graph (Phase 0-bis). (Semantic statement-embeddings = later ablation.)
- **node_text** = theorem/def names, statements (types), docstrings — for keywords + agent context + the LLM-judge.
- **seeds** = the target/broken declarations.

**Phase 2 sanity check (cheap, no agent):** build a curriculum on the mathlib dep graph (a subfield)
and eyeball whether it's a sensible foundations→frontier math curriculum. Validates the lib on a
genuinely new, embedder-free domain before any agent work.

## Phase 3 — task generation + graded ("partial-success") scoring

**Breakage modes** (generate tasks at varying granularity / difficulty):
- **Delete a whole set of theorems** → ask the agent to reconstruct the subtheory.
- **Replace one or more proofs with `sorry`** → ask it to re-prove.
- **Derail proofs** by adding / removing / editing steps → ask it to repair.

**Finer-grained, graded success** (Lean-verified, but *not* coarse pass/fail — the binary per-goal
check aggregates to a continuous score):
- fraction of deleted theorems / `sorry`-ed proofs **closed** (type-checks);
- number of `sorry`s **remaining** vs introduced;
- **partial proof progress** (goals closed / proof-state advancement for incomplete proofs);
- **edit distance / steps-to-fix** for derailed proofs;
- **efficiency**: attempts / tokens / wall-clock to close;
- **reliability**: variance of the above across tasks (and seeds).

This graded scorer is the key deliverable — it lets us see *degrees* of help from the curriculum,
not just a pass-rate cliff.

## Phase 4 — the experiment: in-context curriculum as a *proof decomposition*

**Reframe (away from RAG):** the curriculum here is not a retrieval index, it's a **proof plan /
decomposition**, delivered **fully in context** (no external KB). RAG is only the eventual *scaling*
axis — **in-context → (high-level index in context + retrieve details) → full KB** — and we **start
in-context**. In Lean-extractive mode "the curriculum" is just the dependency cone of the target T,
**ranked (PageRank) and topologically/frontier-ordered, top-N selected** — i.e. *which* and *how
many* prerequisite theorems to surface, in *what order*.

**Setup (matches the delete-a-set breakage mode):** delete T **together with a connected chunk of
its prerequisite sub-DAG**, so the prerequisites aren't available to just cite — there's something to
break up. The curriculum gives the **bottom-up reconstruction order**. The *granularity* — how far
down the cone you reconstruct before "just assume it" — is the curriculum's selection knob (the
top-N / depth cut).

**No persistent curriculum object.** Selection+order is computed *transiently* to scaffold T, then
discarded — nothing is stored or retrieved.

**A true curriculum is a *selected subset*, not all subproblems.** Proving the whole prerequisite
cone would likely cost *more* than monolithic; a curriculum picks a small set of **high-value,
load-bearing lemmas that build up** to make T tractable. So the arms separate **three** effects —
context, decomposition, and (the real prize) **selection**:
- **Control** — prove T monolithically, no extra context. (base time / tokens / success)
- **T1 — context only:** monolithic, with the selected prerequisite **statements** in context.
  *Does merely seeing the lemmas help?*
- **T2-all (naive decomposition):** prove the **whole** prerequisite cone first, then T — the
  *expensive strawman*. Its job is to show decomposition *alone* isn't the win.
- **T2-curriculum:** prove only the **curriculum-selected subset** of load-bearing lemmas (seeded-
  PageRank importance, optionally spectral-diversity-spread — Phase 0-bis), then T, showing the
  student its earlier proofs. **Hypothesis: T2-curriculum < both control and T2-all.** If
  T2-all ≥ control, that proves **selection** — *which* lemmas to pre-prove — not mere decomposition,
  is what the curriculum buys. Selection may be **recursive** (a selected lemma itself decomposed).

Compare **summed** time/tokens/success over all proved subproblems + T.

**Context ablation inside T2** (a scaffolding spectrum; (a) is the most "curriculum"-like — cumulative
learning from its own worked solutions):
- (a) each subproblem-proof sees its **own proofs** of earlier subproblems;
- (b) sees only the **statements** of earlier subproblems (citable lemmas, not their proofs);
- (c) sees only the **immediate prerequisites as assumptions** (given hypotheses).

**Metric (Phase-3 graded scorer):** headline axis is likely **success on hard theorems** — the win
case is "monolith fails outright, decomposition succeeds." Time/tokens are *efficiency given success*,
and must **count the decomposition overhead** (subproblems are extra proving). The claim holds if
either (i) decomposition succeeds where monolith fails, or (ii) the summed cost is lower because each
subproblem is much easier. Report reliability (variance across tasks/seeds).

Plug into an existing prover/agent loop (LeanDojo/ReProver) or a thin agent harness.

On completion, the Lean work (Phases 2–4) **splits to `~/src/curriculum_lean/`**; the library stays
in `~/src/curriculum`.

## Phase 5 (later) — Rust codebases (second target)

Same shape, harder on every axis: function/type **code-dependency graph** needs static-analysis
tooling (rust-analyzer / MIR call graph); **verification = test suite / compilation** (noisier,
coverage-dependent); synthetic-bug **injection** (mutate code, require tests to pass) is the clean
task source (natural-bug corpora are sparse for Rust); a call graph is a **fuzzier prerequisite
proxy** than a proof DAG. Do it only after Lean proves the method.

## Honest risk read
- **Extraction is the easy part** (algorithms done + validated graph-generic); **the agent-eval
  harness (Phases 3–4) is the real project.**
- Pure-graph mode removes the embedder dependency for Lean (big simplification) — but the *clustering*
  then leans entirely on the dependency graph's community structure; verify it produces sensible
  topics in the Phase-2 sanity check before building the eval.
- "Curriculum as context" must be a *selected slice* (prerequisites of the target), not the whole
  curriculum (context-length). Designing that selection is part of Phase 4.

## Status / working notes
- **Phase 1 + Phase 0-bis DONE** (2026-05-31). Extracted to a standalone `curriculum/` package
  (top-level, parallel to `bookgap/`). Fully decoupled: `grep -rn "import bookgap\|from bookgap"
  curriculum/core/` is empty, and there are zero real bookgap import statements anywhere in the
  package (core imports only stdlib + numpy/scipy/scikit-learn; `ollama` is lazy-imported inside the
  judge layer). mypy-clean (`python -m mypy curriculum/ --ignore-missing-imports`), 49 tests pass.
  - **Layout.** `graph_source.py` (`GraphSource` Protocol + `InMemoryGraphSource`), `config.py`
    (`CurriculumConfig` with rank/select/cluster/order + embeddings_mode + layer toggles +
    diffusion), `build.py` (`build_curriculum`). `core/` = minimal core (pagerank, dendrogram, rank,
    order, circuitrank). `layers/` = optional (lpp incl. pure-graph spectral, clustering=community
    detect, keywords, diffusion=GDC, quality basket, judge=ollama). `tests/` = unit + parity gate.
  - **Parity gate PASSED.** `curriculum/tests/test_parity_wiki.py` loads the cached subgraph
    `cache/wiki/store/enwiki_full/curriculum_subgraph_n40000.npz` as a plain `InMemoryGraphSource`
    (no bookgap) and reproduces the committed `docs/wikipedia-curriculum.md` byte-for-byte: 12
    topics, every per-rank cluster size, every top-5 representative page, every headline keyword.
    The committed doc was made with the demo's DEFAULTS (raw-PR-mass selection + MASS order, NOT
    bg-norm/frontier/LPP — those are the configurable knobs, validated separately in the doc's
    sibling experiments); the gate matches what is actually committed.
  - **Pure-graph path (Phase 0-bis) is first-class.** `embeddings_mode=spectral_from_graph` derives
    geometry from the adjacency alone via `scipy.sparse.linalg.eigsh` on the hub-suppressed
    normalized Laplacian (identity-feature link-LPP; no dense `X L Xᵀ`); `community_detect` clusters
    the graph directly. Both tested with `node_embeddings() is None`.
  - **For the Lean adapter (Phases 2–4):** implement the `GraphSource` Protocol over the proof-dep
    graph (`adjacency` directed `i->j` = "decl i depends on premise j"; `seed_nodes` = the target
    decls; `node_text` = decl names/statements; `node_embeddings` returns `None`). Set
    `embeddings_mode=SPECTRAL_FROM_GRAPH` (or `COMMUNITY_DETECT`), `ordering=FRONTIER_ADJACENCY`. For
    the extractive "selected subset, ordered" use, set `clustering.enabled=False` to get the bare
    rank->select->order core (each selected node is its own "topic"); `Curriculum.topic_order` then
    gives the prerequisite->frontier reconstruction order over the selected decls.
- **Phase 2 DONE** (2026-05-31). Built `curriculum_lean/` (top-level, parallel to `curriculum/` and
  `bookgap/`): `lean_graph_source.py` (`load_lean_graph` -> `LeanGraph` over the
  `cache/lean/decl_graph/` TSVs, ~9.5 s / ~100 MB, CSR, name<->idx), `decompose.py`
  (`decompose_target` = the extractive bg-norm-select + frontier-order core; `prerequisite_cone`
  forward BFS), `sanity_check.py` (driver). mypy-clean, 5 unit tests pass (tiny synthetic graph).
  Results: `docs/lean-curriculum-sanity.md`.
  - **Pure-graph sanity check PASSED.** Four recognizable targets (Lagrange, quadratic reciprocity,
    Wilson, IVT). Orientation is **correct** (foundations early, the target's immediate premises last
    — e.g. IVT ends on `IsPreconnected`/`ContinuousOn`/`Set.Icc`; Wilson surfaces
    `FiniteField.prod_univ_units_id_eq_neg_one` + `Finset.prod_Ico_id_eq_factorial`). Selected sets
    read as real proof sketches. **bg-norm is essential** (and the default): raw single-seed forward
    PR is dominated by universal hubs (`Eq`/`Nat`/`outParam`); bg-norm demotes them and surfaces the
    target-specific lemmas. Cycle handling = frontier-adjacency `in-out` degree (no topo sort, no
    condensation) worked. One caveat for Phase 3: mathlib's proof-term graph is heavy on typeclass
    scaffolding (`.mk`/`.to…` projections) — add a `kind` filter for the displayed curriculum.
  - **For Phase 3:** `LeanGraph.with_seeds`/`index_of`/`.names|.kinds|.modules|.statements` map
    decls<->nodes; `prerequisite_cone(graph, target)` gives the delete-and-reconstruct cone AND the
    subgraph to restrict the frontier-adjacency order to (induce `adj[cone][:,cone]` -> sub-second vs
    the ~45 s full-graph order — the recommended optimization); `decompose_target(...).ordered_names()`
    is the bottom-up reconstruction sequence, `top_n` the granularity/depth knob.
- **Phase 3 DONE** (2026-05-31). Added `curriculum_lean/{verify,tasks,score}.py` + a Lean
  metaprogram `curriculum_lean/lean/ExtractSignatures.lean`. Results + Phase-4 interface:
  `docs/lean-eval-harness.md`. mypy-clean (12 files), 22 default tests pass, 10 `integration`
  tests pass (~110 s).
  - **Verification approach (`verify.py`).** Scratch file = `import Mathlib` + preamble +
    spliced candidate; `lake env lean <scratch>` from `~/src/mathlib4` against the olean cache
    (~3.2 s, ~6.6 GB/check). Output parsed into a graded `VerificationResult`
    (`type_checks`/`sorry_free`/`n_open_goals`/`status` ∈ SUCCESS/SORRY/UNSOLVED_GOALS/
    TYPE_ERROR/PARSE_ERROR/TIMEOUT/OTHER_ERROR). Per-call timeout with `start_new_session` so
    the tree is reaped (never hangs). Classifier unit-tested on captured real lean output.
  - **The crux that made the gate pass.** The decl-graph pp `statement` is *not* reliably
    re-parseable. Re-derive the goal type from the loaded env via a `#sig` metaprogram with
    `pp.coercions.types`/`pp.numericTypes`/`pp.universes` true (without these, coercion targets,
    numeric-literal types, and universe levels are elided and `@<fqn>` fails to type-check). With
    them, `theorem t : <type> := @<fqn>` verifies sorry-free for every sampled target. Ground
    truth = `@<fqn>` (robust; the human source proof is a fragile non-requirement).
  - **VALIDATION GATE PASSED.** sorry-mode, all four Phase-2 sanity targets: (a) ground-truth
    `@<fqn>` → success (~3.2 s), (b) bare `sorry` → not-success (status SORRY). delete-cone
    (Wilson, 5-goal chain): ground-truth reconstructs 100%, sorry 0%. derail: `over_apply` breaks
    (parse_error), ground-truth fix verifies.
  - **Task modes (`tasks.py`):** `make_sorry_task`, `make_delete_cone_task` (kind-filtered
    cone via `prerequisite_cone` + `decompose_target` bottom-up order, target appended),
    `make_derail_task` (`verify_broken=True` keeps the first genuinely-failing perturbation);
    `sample_theorem_targets`, `extract_signatures` (batch `#sig` driver).
  - **Scorer (`score.py`):** `score_task(task, provider)` where a provider is `Goal->proof_str`;
    `TaskScore` (fraction_closed, sorries_remaining, mean_partial_progress, edit_distance,
    placeholders attempts/tokens/wall_clock); `cross_task_variance` -> `BatchScore` (reliability).
    `ground_truth_provider`/`sorry_provider` double as the gate.
  - **Phase 4 hook:** an arm IS a candidate provider `Goal -> proof_string`; see
    `docs/lean-eval-harness.md` "Interface notes for Phase 4".
- (next) Phase 4 (the control/T1/T2 experiment). Also still pending: migrate the `curriculum/`
  lib + this doc to `~/src/curriculum/` and split `curriculum_lean/` (+ Phases 3–4) to
  `~/src/curriculum_lean/`.

## Pointers
- Engine history + validated graph-generic features: `docs/backlog/curriculum-generalization-pedagogical-subspace.md`.
- Quality metrics (intrinsic + extrinsic): `docs/curriculum-quality-metrics.md`.
- Lean ecosystem: LeanDojo + ReProver (traced mathlib + premise retrieval), miniF2F, ProofNet.
