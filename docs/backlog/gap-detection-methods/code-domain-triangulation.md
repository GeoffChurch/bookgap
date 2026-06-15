# Gap detection — triangulating the clean-oracle thesis with a third domain: code

**Status: COMPLETE (2026-06-15) — all 4 spec substrates done: Python, Go (CHA→RTA dial), Rust, Bazel monorepo. Results in `docs/code-gap-triangulation.md`.**
They **revised the thesis**: code patterns with Lean (graph wins, embedding at chance), NOT
intermediate, and the axis is gap **modality** (structural vs semantic-void), not graph cleanness; the
temporal holdout is a clean **third negative** (preferential attachment). **The Go CHA→RTA cleanness
dial is the controlled clincher** — two call graphs of different precision on the *same* corpus, with
the static winner **invariant** to cleanness (M1≈0.90, M_emb≈chance under both CHA and RTA). Infra:
generic `artifacts_scripts/curriculum/gap_recover/` harness (core + lean/python/go/rust/bazel adapters;
Lean reproduced exactly). The graph-wins/embedding-weak verdict holds across **Lean + 4 code
substrates**, fine→coarse granularity, noisy→clean edges (the controlled CHA→RTA dial + Bazel's exact
build deps), and a **4.3× coupling range** (+0.037→+0.161) — modality, not cleanness/granularity/
coupling. **All spec substrates covered; the triangulation is complete.** The full design — staged program,
generic-core refactor, and the Stage-1 (Python) design — is in *## Design / working notes
(2026-06-15)* below.

## The thesis we're triangulating

From two domains we have a two-point line, not yet a law:

> **STATIC gap-detection method ∝ oracle cleanness.** When the dependency graph is a *clean
> oracle*, a graph-citation-ratio detector is sharp and wins; when it is noisy, the graph signal
> fails and an embedding density-ratio is needed instead.
>
> **TEMPORAL prediction is a universal negative.** A detected gap does *not* forecast where the
> field's next unit lands — preferential attachment does — independent of oracle cleanness.

The two points (see `docs/end-to-end-overview.md` Stage 1 + "holes", and memory `lean-gap-detection`):

| domain | oracle | static winner | temporal |
|---|---|---|---|
| **articles** (OpenAlex/Wikipedia) | citation graph is **noisy** | embedding density-ratio / level-set | negative (reroute ≠ arrival) |
| **Lean/Mathlib** | proof-term `uses` DAG is a **clean oracle** | graph citation-ratio Δ = log(def_users+1) − log(thm_users+1) | negative (preferential attachment) |

Two points define a line through "cleanness." **Code is the third point that tests whether it's
actually a line** — and, with the right substrate spread, whether *granularity* is a confound.

## Why code, and the prediction

A code dependency graph (calls / imports / build edges) is an *intermediate* oracle: more explicit
than article citations, fuzzier than a proof DAG (dynamic dispatch, generics/traits, reflection,
incomplete static resolution all inject noise). So the thesis makes a **falsifiable prediction**:

- **Static:** graph-citation gap-detection should win *partially* — sharper than in articles,
  softer than in Lean — and should **degrade monotonically as the call graph gets noisier** across
  the substrates below. If a noisy code oracle instead behaves like articles (embedding wins), the
  cleanness axis is confirmed from the noisy side too.
- **Temporal:** still negative. New code should land by preferential attachment (already-central
  functions/modules grow), not in the graph's gaps — mirroring both prior domains.

## The F / B analogy for code

The Lean split was **B = definitions** (concepts) vs **F = theorems** (elaborations that use them);
the gap = a concept structurally embedded but under-theorized (high def-usage, low theorem-usage).
The code analogue (the substrate-specific modeling choice):

- **B = types / data structures / public API surface** (the "concepts" — things composed into
  other things).
- **F = functions / methods / call sites / tests** (the "operations" that exercise the concepts).
- **Gap = a definition structurally embedded (composed into other types, imported widely) but
  under-exercised** (few functions act on it, few tests cover it) relative to its graph neighbors —
  the "under-theorized concept" analogue. Δ = log(structural-users+1) − log(operational-users+1).

The harness is substrate-generic enough that the exact split can be tuned per substrate; the two
*evaluations* below only need (graph, node-text, a notion of usage-growth).

## Method — reuse the Lean harness, swap the substrate

Both validated Lean evaluations are substrate-generic and are the template
(`artifacts_scripts/curriculum/lean_gap/`):

1. **Plant-and-recover** (`run_lean_gap_plant_recover.py`): delete a concept's cluster of
   usages/tests (the code analogue: delete a module's test suite or a function's call sites), then
   ask whether the detector recovers the orphaned region (defs whose usage-ratio grew ≥ threshold).
   AUC = MannWhitney(region vs control), **null-calibrated against LOCAL deletions** (the
   de-confounding null was load-bearing in Lean — keep it). Clustered bootstrap over modules.
2. **Temporal holdout** (`run_lean_gap_temporal_holdout.py`): code has rich git history. At tag T,
   compute the detector; score against which defs actually gain new functions/tests/usages by
   T+Δ. **Growth-matched null on deciles of (existing-usage-count, in-degree)** to strip the
   centrality confound — this is what exposed preferential attachment in Lean.

Reuse `lean_gap_kernels.py` (the Δ / density kernels), `build_lean_gap_data.py` (swap the graph
loader), `build_lean_gap_births.py` + `fqn_reconstruct.py` (temporal name-resolution analogue =
resolve symbols across tags). Keep transcript hygiene: graphs, ids, counts — never source bodies.

## Substrate options (a cleanness × granularity spread)

Originally three; the 2026-06-15 design adds **Go** (option 4 below) and runs them as a staged
program in increasing difficulty — see *Design / working notes*. The value compounds as more are
placed on the axis.

1. **Python import + call graph** — *cheapest, fuzziest oracle, fine granularity.* Static call
   resolution via `ast` + `jedi`/`pyan`/`pydeps` (or a project's own import graph). Noisy:
   duck-typing, dynamic dispatch, monkey-patching, reflection, incomplete static resolution →
   sits **closest to the article (noisy) end.** The sharpest test of "does a noisy code oracle
   behave like articles?" Task source for plant-and-recover: delete a package's `tests/`. Cheap,
   huge corpus (PyPI / any large repo), no build needed.
2. **Rust `rust-analyzer` / MIR call graph** — *cleaner static resolution, fine granularity*
   (matches the existing note in `curriculum-standalone-extraction-and-lean.md` §Phase 5). Strong
   types + mostly-static trait resolution → a **sharp-ish call graph** (some noise from `dyn`
   trait objects + generics); sits **mid-to-clean.** Verification = test-suite / compilation;
   synthetic-bug **injection** (mutation, require tests to pass) is the clean task source. Most
   tooling cost of the three.
3. **Large monorepo module / build-dependency graph** — *cleanest edges, coarse granularity, big
   scale.* Bazel/Buck/Pants build graphs (or a large org import graph) encode **exact** module
   dependencies → a **clean oracle at the package level**, millions of edges. Coarse (module, not
   function), so it **crosses cleanness with granularity**: if graph-citation wins here too, clean
   edges suffice; if it degrades, granularity matters independently of cleanness — a bonus axis the
   other three can't isolate.
4. **Go `go/callgraph` (CHA / RTA / VTA)** — *low–moderate cost, fine granularity, and a built-in
   within-corpus cleanness dial.* The 2026-06-15 **centerpiece** (full detail in *Design / working
   notes*): the same corpus run under `cha` (over-approximate, noisy) vs `rta`/`vta` (precise,
   clean) is a *controlled* cleanness contrast — domain, embedder, granularity and scale all held
   fixed — so it tests the line causally, not as a third confounded dot. Off-the-shelf
   `golang.org/x/tools`; `_test.go` is a clean plant site; huge corpus + rich git history.

## Design / working notes (2026-06-15)

Decided in a brainstorming session. Supersedes nothing above — it *operationalizes* the substrate
options into a staged program with a shared harness.

### Shape: a staged program, easiest substrate first

Not one confounded third dot — a *spread* that traces the cleanness line. Run substrates in
increasing difficulty; each adds a row to the thesis table:

1. **Python** — *noisy / fine granularity.* `ast` import graph (exact) + call graph via
   `jedi`/`pyan` (noisy: duck-typing, dynamic dispatch). Plant site: delete a module's `tests/`.
   **Buys:** "does a noisy code oracle behave like articles?" — the noisy-end anchor (expect the
   embedding density-ratio ≈ wins, graph-citation weak).
2. **Go** — *low–moderate cost; the CENTERPIECE.* `golang.org/x/tools/go/callgraph` ships
   **CHA → RTA → VTA** (decreasing noise) on the *same corpus*, so this stage is a **controlled
   cleanness dial**: rerun the identical corpus under `cha` (noisy) vs `rta` (clean) and watch the
   static winner flip with domain/embedder/granularity/scale held fixed. This is the causal test
   articles-vs-Lean can't give. Plant site: delete `_test.go`. F/B: types/interfaces vs
   funcs/methods + `Test*`. A small Go extractor emits the graph as TSV; the Python harness ingests.
3. **Rust** — *clean / fine; HARDEST.* `rust-analyzer`/MIR call graph; builds per tag;
   mutation-injection as the clean task source. The clean fixed point.
4. **Monorepo build graph** — *clean / coarse; OPPORTUNISTIC, not strictly ordered.* `bazel query`
   / Buck / Pants edges are exact and trivial to extract — its only difficulty is **finding a
   public monorepo with rich tagged history.** Slot it whenever such a corpus is in hand; it
   uniquely crosses **cleanness × granularity** (the bonus axis the fine-grained three can't isolate).

### Architecture: one generic core, thin per-substrate adapters

`lean_gap_kernels.py`, `run_lean_gap_plant_recover.py`, and `run_lean_gap_temporal_holdout.py` are
*already* substrate-agnostic — they only touch `(node-text embeddings, F/B masks, edge list,
structural/operational user counts, cross-tag births)`. So:

- **Promote** `artifacts_scripts/curriculum/lean_gap/` → a substrate-neutral
  `gap_recover/core/` (kernels + both evaluators + bootstrap/null machinery), and **migrate Lean to
  be its first adapter** (`gap_recover/adapters/lean/`). Per the no-back-compat rule, this is a
  migration: delete the old `lean_gap/` path and update imports — **no dual-path**.
- Each substrate is then **only an adapter**: a graph extractor that emits the shared `data.npz`
  contract; nothing downstream changes.

**Shared `data.npz` contract** (generalized from `build_lean_gap_data.py`):
`name, kind, module, text, edges(src→dst = depends-on), is_F, is_B, structural_users,
operational_users`, plus the gated `subgraph.npz` (CSR) and git-tag `births` for the temporal arm.

**F/B analogy:** `B` = types / structs / interfaces (concepts); `F` = functions / methods + tests
(operations). `Δ = log(structural_users + 1) − log(operational_users + 1)`.

### Stage-1 (Python) defaults

- **Corpus:** a curated ~20–40 large, well-tested PyPI packages (many bootstrap units + diversity
  for the clustered bootstrap), each with real `tests/` and tagged history. (Alt considered: one big
  multi-module library, e.g. scikit-learn/Django — rejected for fewer bootstrap units.)
- **Node-text embedder:** reuse **bge-small** (same as Lean) so the embedding-detector arm is
  apples-to-apples across domains; add a code embedder (`jina-embeddings-v2-base-code`) only as an
  optional robustness check.
- **Plant site:** delete a module's `tests/` (the operations exercising a concept) → recover the
  now-under-exercised types — the direct analogue of deleting a concept's theorem cluster.

### Rigor (inherited from Lean — non-negotiable)

Plant-recover: **local-deletion null** (load-bearing in Lean — keep it). Temporal:
**growth-matched null on (existing-usage-count, in-degree) deciles** (this is what exposed
preferential attachment). **Clustered bootstrap** over packages/modules; **two seeds.** Transcript
hygiene: graphs / ids / counts only — **never source bodies.**

### Decomposition into implementation plans

- **Plan 1:** core refactor + Lean adapter migration + **Stage 1 (Python)** end-to-end through both
  evaluations.
- **Plan 2+:** Go (the dial), then Rust; monorepo when a corpus is available. Each its own plan.

## Done / success criteria

- At least one substrate run through both evaluations with the Lean-grade rigor (local-deletion
  null, growth-matched temporal null, clustered bootstrap, two seeds).
- A third row added to the thesis table in `end-to-end-overview.md` placing code on the cleanness
  axis, and a statement of whether the static-method-∝-cleanness line holds (or where it breaks).
- The temporal result reported either way (a third "negative" strengthens universality; a positive
  would be the surprise worth chasing).

## Pointers

- Thesis + the two existing points: `docs/end-to-end-overview.md` (Stage 1, "the map / holes"),
  memory `lean-gap-detection`, memory `gap-book-surprise-definition`.
- Lean harness to reuse: `artifacts_scripts/curriculum/lean_gap/{build_lean_gap_data,lean_gap_kernels,run_lean_gap_plant_recover,run_lean_gap_temporal_holdout,fqn_reconstruct,build_lean_gap_births}.py`.
- The pre-existing code-as-curriculum note (different framing — selection, not gap-detection):
  `docs/backlog/curriculum-standalone-extraction-and-lean.md` §"Phase 5 — Rust codebases".
- Substrate for Lean (the clean-oracle exemplar): `docs/lean_decl_graph.md`.
