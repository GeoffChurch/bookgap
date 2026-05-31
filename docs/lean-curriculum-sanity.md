# Lean extractive-curriculum sanity check (Phase 2)

Phase 2 of `docs/backlog/curriculum-standalone-extraction-and-lean.md`: a
pure-graph, no-agent eyeball of the extractive curriculum over the mathlib4
declaration-dependency graph (`docs/lean_decl_graph.md`). The question: given a
**target decl**, does seeded-PageRank-select + frontier-adjacency-order produce a
sensible **foundations → target** prerequisite ordering? Verdict: **yes** — the
orientation is correct (primitives/algebra early, the target's immediate premises
last) and the selected sets read as real proof sketches, with one honest caveat
about typeclass-scaffolding density (below).

## Setup

- **Graph** = the full mathlib4 v4.30.0 decl graph: **466,633 nodes**,
  **12,619,118 directed edges** (`src → dst` = "decl `src` depends on premise
  `dst`"), loaded into `curriculum.InMemoryGraphSource` via
  `curriculum_lean.load_lean_graph` (streams the gzipped TSVs straight into a
  CSR; ~9.5 s, ~100 MB).
- **Engine** = the decoupled `curriculum/` lib (commit `49233de`), extractive
  minimal core: `RankMethod.PAGE_RANK`, `clustering.enabled=False` (each selected
  node is its own topic), `keywords/quality/judge` off,
  `embeddings_mode=COMMUNITY_DETECT` purely so no embeddings are resolved (no
  `eigsh` is ever run on the full graph), `ordering=FRONTIER_ADJACENCY`
  (cycle-tolerant). α=0.85.
- **Seeds** = `[target]` (single decl).
- **Selection** = **background-normalized** (`p^a · log((p+ε)/(g+ε))`, a=0.5),
  where `p` is the target-seeded forward PR and `g` is a uniform-teleport
  background PR on the same adjacency. This is the load-bearing knob — see the
  raw-mass comparison below. **top-N = 40**, target excluded from its own list.
- Driver: `python -m curriculum_lean.sanity_check` (per-target JSON at
  `cache/lean/sanity_{bgnorm,rawmass}.json`). ~57 s/target (the cost is the
  41-batch absorbing PageRank in the frontier-adjacency order, not the seeded
  rank).

## Sample targets (different areas)

| Target decl | Theorem | Area |
| :-- | :-- | :-- |
| `Subgroup.card_subgroup_dvd_card` | Lagrange's theorem (`Nat.card ↥s ∣ Nat.card α`) | group theory |
| `legendreSym.quadratic_reciprocity` | Quadratic reciprocity | number theory |
| `ZMod.wilsons_lemma` | Wilson's lemma (`↑(p-1)! = -1`) | number theory |
| `IsPreconnected.intermediate_value` | Intermediate value theorem | analysis / topology |

## Verdict on orientation: correct (foundations → target)

For every target, the most foundational primitives sort first and the target's
**immediate premises** sort last — the literal frontier. Concretely, the last
ranks of each curriculum are the decls a mathematician would name as the direct
inputs to the proof:

- **Lagrange** (40 selected). Early: `Mul`, `Dvd`, `Group`/`Semigroup`/
  `CommSemigroup` and their constructors. Frontier: rank 30
  `Subgroup.card_eq_card_quotient_mul_card_subgroup` (the `|G| = |G/s|·|s|`
  lemma Lagrange is one line from), rank 37 `dvd_mul_left`, rank 38 `Subgroup`,
  rank 40 `Dvd.dvd`. Exactly the immediate logical premises.
- **Quadratic reciprocity** (40). Early: `Fact`, `Mul`, `One`, `ZMod`. Frontier:
  `quadraticChar_odd_prime`, `quadraticChar_neg_one`, `quadraticChar_sq_one`,
  `ZMod.χ₄`, `quadraticChar`, and rank 39 `legendreSym` (the very object in the
  target). This is the actual mathlib proof route — Gauss-sum / quadratic-
  character machinery — surfaced as a sketch.
- **Wilson** (40). The selected set reads as a proof outline: `Finset.prod_Ico_id_eq_factorial`
  ((p−1)! as a product), `FiniteField.prod_univ_units_id_eq_neg_one` (product of
  all units of a finite field = −1 — the heart of the proof),
  `Units.instCommGroupUnits`, `ZMod.instField`, `Finset.prod_bij` (reindexing),
  with order/field/prime primitives early.
- **IVT** (40). Early: order theory (`LE`, `Preorder`, `PartialOrder`, `Lattice`,
  `LinearOrder`, `DistribLattice`), `Set`/`HasSubset`, `TopologicalSpace`.
  Frontier: rank 23 `IsPreconnected`, rank 24 `Set.image`, rank 26
  `IsPreconnected.intermediate_value₂` (the binary helper the main theorem
  directly invokes), rank 27 `ContinuousOn`, rank 34 `Set.Icc` — i.e. the
  hypotheses and conclusion of the statement
  `IsPreconnected s → ... ContinuousOn f s → Set.Icc (f a) (f b) ⊆ f '' s`.

So: **orientation is right, not reversed.** Forward PR seeded at the target flows
decl → premise and accumulates on foundations; frontier-adjacency then puts
high-(in−out) (depended-on, depends-on-little) topics first.

## The load-bearing knob: background normalization

Single-seed forward PR over this graph is dominated by **graph-universal
primitives** that are prerequisites of *everything* and carry zero target-specific
signal. Raw-mass top-6 per target makes the pathology obvious:

| Target | raw-mass top-6 |
| :-- | :-- |
| Lagrange | `outParam, Eq, Mul, Nat, Dvd, Group.mk` |
| Quad. recip. | `Nat, outParam, Mul, Eq, Semiring.mk, Eq.refl` |
| Wilson | `Nat, Monoid.mk, One, Mul, outParam, Eq` |
| IVT | `LE, Set, outParam, And, HasSubset, LinearOrder.mk` |

`Eq`, `Nat`, `outParam`, `True`, `Eq.refl` are the global PageRank hubs (highest
in-degree in the whole graph), not lemmas anyone would teach for *this* theorem.
**Background normalization demotes exactly these** — `Eq`/`Nat`/`outParam` fall
out of the bg-norm top lists entirely — and promotes the target-specific
`ZMod`/`quadraticChar`/`Subgroup.card_eq_…`/`IsPreconnected` decls shown above.
This matches the prediction in the backlog ("identity-free, demotes globally-
popular hubs") and the wiki-side result in `docs/wikipedia-curriculum-bgnorm.md`.
**Recommendation: bg-norm is the default for the extractive Lean use.**

## Honest caveat: typeclass-scaffolding density

mathlib's dependency graph is a *proof-term* graph: every theorem's deps include
the **typeclass instances and structure projections** its proof term mentions
(`Group.mk`, `CommMagma.toMul`, `SemilatticeInf.toPartialOrder`, `…instField`,
`…instMembership`, …). These are genuine premises, so they correctly appear — but
they make the selected set heavier on algebraic *plumbing* than a human syllabus
would be, especially for results whose statement carries many instances (Lagrange
and IVT are the worst; quadratic reciprocity and Wilson, whose proofs lean on
named domain lemmas, read most like a curriculum). This is **not** an orientation
bug — the plumbing sorts to the foundations, where it belongs — it is a
characteristic of the corpus to manage in Phase 3 (e.g. filter to
`kind ∈ {theorem}` for the *displayed* curriculum, or keep `def`/`ctor` only as
ordering context). Worth an explicit kind-filter knob there; not needed for the
orientation check.

## Loading / scale story

- Streaming load of both gzipped TSVs into a CSR is ~9.5 s and ~100 MB resident
  (32-bit index buffers via `array('i')`, one `csr_matrix((data,(rows,cols)))`
  build, `sum_duplicates()`). No dense matrix is ever formed.
- The seeded forward PR and the uniform background PR are each a handful of
  sparse matvecs over 12.6 M edges — a few seconds, fine at full scale.
- **The cost is the order step.** `compute_frontier_adjacency` runs a batched
  absorbing PageRank with one column per selected node (here 41) over the full
  466 k-node graph: a 466 k × 41 dense float64 iterate (~150 MB) for ~85
  iterations, ≈ 45 s. That dominates the ~57 s/target. It is correct and
  cycle-tolerant, but for Phase 3 it should be restricted to the relevant
  subgraph (see interface notes).
- **Cycle handling that worked.** The graph is not strictly acyclic
  (`brecOn`/`below` recursor machinery, mutual inductives). We never call a
  topological sort. Frontier-adjacency ordering (`in-degree − out-degree` over
  the absorbing-PR inter-topic flow) tolerates cycles by construction and
  produced sane orderings on all four targets — no condensation needed.
- The ~195 sinks (15 kernel axioms + primitive leaves) are legitimate
  foundations; empty-statement nodes fall back to `name` in `node_text` (the
  loader handles this; none of the four targets hit it).

## Interface notes for Phase 3 (task generation)

- **decl ↔ node.** `LeanGraph.index_of(name)` / `.has(name)` /
  `.names[idx]` (plus `.kinds`, `.modules`, `.statements`). `with_seeds([names])`
  returns a fresh `InMemoryGraphSource` (frozen; shares the arrays, no copy) with
  the chosen seeds — that is how you point the engine at any target/broken decl.
- **target + prerequisite cone.** `curriculum_lean.decompose.prerequisite_cone(graph,
  target, max_depth=…, max_nodes=…)` does the forward BFS over `src → dst` and
  returns the dependency-cone node indices (the target + everything it
  transitively needs). This is the set to **delete-and-reconstruct**: delete the
  target together with a connected chunk of this cone, then the extractive
  decomposition gives the bottom-up reconstruction order. The cone also bounds
  the subgraph to restrict the frontier-adjacency order to (induce
  `adj[cone][:, cone]`, re-run `order_by_frontier_adjacency` over it) — that turns
  the ~45 s full-graph ordering into a sub-second one and is the recommended
  Phase-3 optimization.
- **selection = the curriculum.** `decompose_target(graph, target, top_n=…)`
  returns a `Decomposition` whose `.prerequisites` is the ordered (foundations →
  frontier) list of `PrereqDecl(rank, name, kind, module, statement,
  selection_score, pagerank_mass)`; `.ordered_names()` is the bottom-up
  reconstruction sequence. `top_n` is the granularity / depth knob from Phase 4.
- **kind filter.** Add a `kind`-filter (drop `ctor`/`def` projections from the
  *displayed/selected* curriculum while keeping them for ordering context) to
  counter the typeclass-scaffolding density noted above. Statements are present
  for the in-context prompt; 22 % of nodes also carry a docstring (load with
  `text_mode="full"`).
- **selection mode.** Use `background_normalized=True` (default). Raw mass returns
  universal hubs and is only useful as a contrast.

## Files

- `curriculum_lean/lean_graph_source.py` — loader (`load_lean_graph`, `LeanGraph`).
- `curriculum_lean/decompose.py` — `decompose_target`, `prerequisite_cone`,
  `extractive_config`.
- `curriculum_lean/sanity_check.py` — this experiment's driver.
- `curriculum_lean/tests/` — loader + orientation unit tests (tiny synthetic
  graph; no mathlib).
- `cache/lean/sanity_{bgnorm,rawmass}.json` — full per-target results (gitignored).
