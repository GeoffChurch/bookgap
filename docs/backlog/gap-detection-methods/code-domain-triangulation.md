# Gap detection — triangulating the clean-oracle thesis with a third domain: code

**Status: not started (spec only).** Pickup-ready. The idea lives, until this doc, only in the
2026-06-15 strategy thread.

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

## Three substrate options (a cleanness × granularity spread)

Run one to start; the value compounds if 2–3 are placed on the axis.

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
   other two can't isolate.

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
