# mathlib4 declaration-dependency graph (the `curriculum_lean` GraphSource artifact)

This is the static, pre-extracted input that the Lean `GraphSource` (Phase 2 of
`docs/backlog/curriculum-standalone-extraction-and-lean.md`) consumes: a
**declaration → premise** dependency graph over mathlib4, plus per-node text
(name / statement / docstring). Embeddings are *not* part of the artifact — the
Lean curriculum runs in pure-graph mode (identity → spectral-from-graph), so the
graph alone carries all meaning.

> **Version note (2026-05-31).** The graph was re-extracted from mathlib4
> **v4.20.0** (`c211948581bde9846a99e32d97a03f0d5307c31e`) — down from v4.30.0 —
> so it stays coherent with the LeanDojo interactive trace, which only works on
> v4.20.0 (LeanDojo 4.20.0's bundled metaprograms break on the Lean 4.25–4.27
> String/AST overhaul; see `docs/lean-eval-harness.md`). The Size section below
> is the re-extracted v4.20.0 graph; the v4.30.0
> graph is preserved at `cache/lean/decl_graph_v4.30.0/` because it pairs with
> the v4.30.0 Goedel whole-proof verifier (the version split is the one
> follow-up caveat — a fully-coherent whole-proof-vs-interactive comparison
> would re-run Goedel on v4.20.0).

## Size (v4.20.0, re-extracted 2026-05-31)

- **1,199,831** constants in the loaded environment; **357,486** kept as nodes,
  **842,345** dropped as compiler-internal.
- **9,643,363** directed edges (mean ~27 deps/decl). Self-loops dropped.
- Node kinds: 239,240 `theorem`, 98,084 `def`, 7,287 `ctor`, 5,352 `recursor`,
  5,250 `inductive`, 2,256 `opaque`, **13 `axiom`** (the Lean/mathlib kernel
  axioms — two fewer than v4.30.0's 15, reflecting the older kernel), 4 `quot`.
  24% of nodes carry a docstring.
- Gzipped on disk: `nodes.tsv.gz` ~15.3 MB, `edges.tsv.gz` ~30.1 MB.
- Extraction wall-clock: ~33 min (single-threaded; the cost is `ppExpr` per type).
  `lake exe cache get` for the oleans: ~55 s (6772 files).

The graph is smaller than the prior v4.30.0 extraction (466,633 nodes /
12,619,118 edges) simply because v4.20.0 mathlib has fewer declarations than
v4.30.0; this is expected, not a regression.

Sanity-check of the depends-on direction: the highest in-degree (most-referenced)
nodes are exactly the foundational primitives — `Eq` (205,776 refs),
`OfNat.ofNat` (97,396), `congrArg` (94,508), `Nat` (86,100), `id` (71,174) —
which is where forward PageRank mass should accumulate.

## What's on disk (not in git)

`cache/lean/` is gitignored (the whole `cache/` tree is). The toolchain, the
6.9 GB mathlib checkout, and the graph artifact all live outside version
control; only the extraction script (`artifacts_scripts/lean/ExtractDeclGraph.lean`)
and this note are committed.

```
~/src/mathlib4_v4.20.0/                  # mathlib4 checkout @ v4.20.0 (decl-graph + Dojo trace)
~/src/mathlib4/                          # mathlib4 checkout @ v4.30.0 (Goedel whole-proof verifier)
cache/lean/decl_graph/nodes.tsv.gz       # node table (v4.20.0)
cache/lean/decl_graph/edges.tsv.gz       # edge list (v4.20.0)
cache/lean/decl_graph_v4.30.0/           # preserved v4.30.0 graph (pairs with Goedel side)
cache/lean/setup_v4.20.0.log             # elan + cache-get sidecar (v4.20.0)
cache/lean/extract_v4.20.0.log           # extraction sidecar (v4.20.0)
cache/lean/run_cache_get_v4.20.0.sh      # lake exe cache get wrapper (tracked job)
cache/lean/run_extract_v4.20.0.sh        # extraction wrapper (tracked job)
```

## Toolchain / provenance

- **elan** 4.2.2; **Lake** 5.x (`~/.elan/bin` on PATH); the toolchain is pinned
  per-checkout by each repo's `lean-toolchain`.
- **mathlib4** checked out at the stable tag **`v4.20.0`** (commit
  `c211948581bde9846a99e32d97a03f0d5307c31e`), pinned toolchain
  `leanprover/lean4:v4.20.0` — the version LeanDojo 4.20.0 itself pins (its repo
  `lean-toolchain`), chosen so the decl-graph and the LeanDojo interactive
  `Dojo` trace stay on one mathlib version. (Was v4.30.0 / `c5ea00351`; downgraded
  2026-05-31 because LeanDojo 4.20.0 cannot trace v4.30.0 — its bundled
  metaprograms break on the Lean 4.25–4.27 String/AST overhaul; full diagnosis
  in `docs/lean-eval-harness.md`.)
- The oleans came from `lake exe cache get` — **no source compile**.
  `import Mathlib` elaborates in ~3 s against the cache.
- **Version split caveat.** The Goedel whole-proof verifier still uses the
  v4.30.0 checkout (`~/src/mathlib4`) and the preserved v4.30.0 graph. So the
  decl-graph + interactive Dojo arm are coherent on v4.20.0, but a
  fully-coherent whole-proof-vs-interactive comparison would require re-running
  the Goedel side on v4.20.0 (follow-up, not done here).

## Schema

Two TSV files, gzipped. Records are one-per-line; the text fields
(`statement`, `doc`, `module`, `name`) are backslash-escaped so a literal tab,
newline, carriage-return, or backslash becomes `\t`, `\n`, `\r`, `\\`
respectively. Decode by reversing those four escapes.

### `nodes.tsv.gz` — node table

| col | field       | meaning                                                            |
|-----|-------------|--------------------------------------------------------------------|
| 0   | `idx`       | 0-based node index; equals the line number. The edge list refers to these. |
| 1   | `name`      | fully-qualified declaration name, e.g. `Nat.add_comm`.             |
| 2   | `kind`      | one of `axiom`, `def`, `theorem`, `opaque`, `quot`, `inductive`, `ctor`, `recursor`. |
| 3   | `module`    | the mathlib/Lean module the decl was defined in, e.g. `Mathlib.Algebra.Group.Defs`. |
| 4   | `statement` | pretty-printed type (the theorem/def signature). May be empty if pretty-printing raised. |
| 5   | `doc`       | docstring if present, else empty.                                  |

Nodes are sorted by `name` for reproducibility, so `idx` is stable across reruns
(modulo a mathlib version bump).

### `edges.tsv.gz` — directed edge list

`src_idx \t dst_idx`. **`src → dst` means "declaration `src` depends on premise
`dst`"** — i.e. `dst` appears in the type or value of `src`. This is exactly the
`GraphSource.adjacency()` convention (`A[i,j] > 0` ⟺ `i` points to `j`,
documented as "decl `i` depends on premise `j`" for the Lean DAG). Forward
PageRank mass therefore flows decl → premise and accumulates on foundational
declarations. Self-loops are dropped.

## How deps are collected (and the internal-name filter)

For each declaration we take `ConstantInfo.getUsedConstantsAsSet`, which folds
over the decl's **type and value** and, for `inductive`/`ctor`/`recursor`, adds
the structurally-implied names (constructors, the family `all`, etc.). So a
theorem's deps are the constants used in its statement *and its proof term* —
every premise the proof actually invokes, which is what makes this a real
prerequisite DAG rather than a "stated-in-terms-of" graph.

**Filter.** Compiler-internal / auto-generated declarations are dropped as
nodes, using `Lean.Name.isInternalDetail` (names with a `_`-prefixed component,
or `match_…` / `proof_…` / `eq_…` / `omega_…` auxiliaries, or any numeric
component). Edges are then emitted only between surviving nodes.

Consequences the GraphSource phase should know:

- A proof's dependency on a premise reached *only* through an internal helper
  (e.g. `theorem.proof_1`, `Foo.match_1`) is **not** transitively re-expanded —
  the edge is simply absent, because we don't walk into dropped nodes. Recorded
  deps are the human-meaningful premises, not the elaborator's internal scaffolding.
  If a fuller transitive premise set is ever needed, drop the filter (keep the
  internal nodes) and let downstream code collapse them, or follow internal-node
  edges one extra hop during extraction.
- `kind = axiom` nodes are the genuine Lean/mathlib kernel axioms (only 15:
  `Classical.choice`, `propext`, `Quot.sound`, `Lean.ofReduceBool`, etc.). They
  are legitimate graph sinks (no deps); keep them.
- `module` is empty only for decls with no `getModuleIdxFor?` (shouldn't happen
  for a pure-import environment); treat empty as "unknown module".
- `statement` can be empty if `ppExpr` raised on a pathological type; the node is
  still present, just with no text. Fall back to `name` for such nodes when
  building `node_text`.
- The graph is **not acyclic in general** at the raw level (mutually-recursive
  inductives, `brecOn`/`below` recursor machinery can form small cycles), even
  though the *mathematical* prerequisite relation is a DAG. Downstream ordering
  (frontier-adjacency / CircuitRank) already tolerates cycles, but don't assume
  a strict topological order exists without condensation.

## Loading into a `GraphSource`

`InMemoryGraphSource` (`curriculum/graph_source.py`) is the natural target:

```python
import gzip, numpy as np, scipy.sparse as sp
from curriculum.graph_source import InMemoryGraphSource

def _unesc(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            out.append({"t": "\t", "n": "\n", "r": "\r", "\\": "\\"}.get(nxt, nxt))
            i += 2
        else:
            out.append(c); i += 1
    return "".join(out)

names, texts = [], []
with gzip.open("cache/lean/decl_graph/nodes.tsv.gz", "rt") as f:
    for line in f:
        idx, name, kind, module, stmt, doc = line.rstrip("\n").split("\t")
        names.append(_unesc(name))
        # node_text: statement is usually the most useful; prepend name, append doc
        texts.append(" ".join(t for t in (_unesc(name), _unesc(stmt), _unesc(doc)) if t))
n = len(names)

rows, cols = [], []
with gzip.open("cache/lean/decl_graph/edges.tsv.gz", "rt") as f:
    for line in f:
        s, d = line.split("\t")
        rows.append(int(s)); cols.append(int(d))
adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

src = InMemoryGraphSource(
    adj=adj, id_list=names, text=texts,
    seeds=np.array([], dtype=np.int64),   # target/broken decls, chosen per task
    embeddings=None,                       # pure-graph: spectral_from_graph
)
```

## Reproducing the artifact

```bash
export PATH="$HOME/.elan/bin:$PATH"
cd ~/src/mathlib4
lake exe cache get                                  # prebuilt oleans (no compile)
BOOKGAP_LEAN_OUT=/abs/out lake env lean \
    ~/src/bookgap/artifacts_scripts/lean/ExtractDeclGraph.lean
```

Or via the tracked-job wrappers (`cache/lean/run_cache_get.sh`,
`cache/lean/run_extract.sh`), each launched through `bookgap.jobs run` so the
work survives in a systemd-run cgroup and logs to a flushed sidecar.
