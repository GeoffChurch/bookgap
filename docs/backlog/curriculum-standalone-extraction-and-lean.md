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

## Phase 4 — the experiment (curriculum vs baseline vs no-context)

Framing: **graph-structured curriculum-RAG vs flat vector RAG vs no RAG.**
- **Curriculum arm:** curriculum-*guided* context — given a broken declaration, surface its
  prerequisite lemmas/concepts **in dependency order** (the relevant curriculum slice, not the whole
  thing).
- **Baseline arm:** flat top-k similarity premise retrieval (ReProver-style — already exists).
- **Control arm:** no retrieval / no extra context.
- Plug into an existing prover/agent loop (LeanDojo/ReProver) or a thin agent harness; score with the
  Phase-3 graded metric + efficiency + reliability.

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
- (not started) — Phase 1 extraction is the entry point. First concrete step: define `GraphSource` +
  `CurriculumConfig`, lift the modules listed in Phase 0, parity-gate against a bookgap curriculum.

## Pointers
- Engine history + validated graph-generic features: `docs/backlog/curriculum-generalization-pedagogical-subspace.md`.
- Quality metrics (intrinsic + extrinsic): `docs/curriculum-quality-metrics.md`.
- Lean ecosystem: LeanDojo + ReProver (traced mathlib + premise retrieval), miniF2F, ProofNet.
