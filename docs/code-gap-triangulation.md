# Code-domain gap detection — triangulating the clean-oracle thesis (Stage 1: Python)

*Status: Stages 1–4 complete (Python, Go CHA→RTA dial, Rust, Bazel monorepo build graph), 2026-06-15.
Spec + program: `docs/backlog/gap-detection-methods/code-domain-triangulation.md`. All spec substrates
covered. This doc reports what the four code substrates showed — and how they **revise** the thesis.*

## What we set out to test

Two prior domains gave a two-point line: **articles** (noisy citation graph → an *embedding*
density-ratio detector wins) and **Lean/Mathlib** (clean proof-term graph → a *graph* citation-ratio
detector wins). The pre-registered prediction was that **code**, an *intermediate*-cleanness
call/import graph, would land *between* them — graph-detection winning *partially*, "softer than
Lean, sharper than articles."

**It did not. Code patterns with Lean, not in between — and the reason is not cleanness.**

## The harness (infrastructure deliverable)

The validated Lean plant-and-recover + temporal-holdout harness was refactored into a
**substrate-generic core** (`artifacts_scripts/curriculum/gap_recover/core/`) reading a neutral
`data.npz` contract, with thin per-substrate **adapters** (`adapters/lean/`, `adapters/python/`).
Regression gate: rebuilding Lean through the new core reproduces the validated result **exactly** —
M1 lift over the local null **+0.148** (orig +0.149), M_emb +0.065, level-set +0.000, temporal M1
stratified AUC ≈ 0.50. So any code-side difference is a property of the *substrate*, not the code.

**F/B mapping for code:** `B` = classes (concepts), `F` = functions/methods + tests (operations).
`Δ = log(structural_users+1) − log(operational_users+1)` (a class composed-into many other classes
but exercised by few functions = a structural-but-under-exercised gap). The Δ ranking is face-valid:
the top concepts are `sklearn.base.BaseEstimator` (su=307, ou=5, Δ=+3.94), `tornado.web.RequestHandler`,
`click.types.ParamType` — exactly "central base class, lightly exercised."

**Corpus:** 21 large PyPI packages (size-stratified, blobless clones), **167,470 nodes**
(17,465 classes, 150,005 functions), **199,482** dependency edges, bge-small embeddings of
signatures+first-docstring-line (same embedder as Lean, for an apples-to-apples embedding arm).
The call graph is deliberately a *noisy* static `ast` resolution (duck-typing/dynamic dispatch
dropped) — code is meant to be the noisy-graph point.

## Static result — plant-and-recover

### The Lean method degenerates on code (and this is informative)

Lean's plant deletes a concept's operation cluster and defines the recovered region as the
**embedding orphan core** — candidates whose k-th nearest-foreground distance grew ≥15%. On Python
this yields **2 usable plants out of 160** — the embedding core almost never forms.

The obvious hypothesis — *call graph and text-embedding are decoupled in code* (a function that calls
a class isn't textually similar to it) — is **wrong**. Measured directly, citation↔embedding coupling
is **comparable in both domains**:

| coupling: citer−random cosine lift | frac concepts with citers closer than random |
|---|---|
| **Lean** +0.115 (median +0.122) | 0.98 |
| **Python** +0.108 (median +0.103) | 0.99 |

Callers *are* embedding-near their concept in code, just as Lean theorems are. The orphan core fails
for a **quantitative** reason: code's operational clusters are smaller (median deletion 31 vs Lean 77)
and a concept's *top-k* nearest foreground is only weakly caller-dominated in **both** domains
(top-10 caller fraction 0.12 Python vs 0.21 Lean), so a small deletion rarely moves the k-th distance
by 15%. The embedding-orphan-core method is therefore **Lean-tuned and fragile**, not a clean
cross-domain probe.

### Graph-region variant — the cross-domain comparison

To make the code plant runnable we added `--region graph` (orphaned region = candidates that lost a
substantial share of their operational support) — one parameterized path in the generic core, the
embedding default unchanged. Run on **both** domains (2 seeds, n=40 plants):

| domain · region=graph | M1 (graph) plant AUC | M1 lift / local-null | M_emb (embedding) plant AUC |
|---|---|---|---|
| **Lean** | 0.929 / 0.944 | **+0.10 / +0.15** (CIs exclude 0) | 0.62 / 0.64 (weak) |
| **Python** | 0.961 / 0.968 | local-null *degenerate* | **0.46 / 0.55 (chance)** |

Reading this honestly:

- **The graph detector M1 recovers planted gaps in both** — and in **Lean** it does so with a
  *working* discriminating null, showing a real **+0.13 lift** over generic local deletions (it
  detects concept-coherence, not just "something lost support"). On **Python** the local null
  *degenerates*: a random localized function-deletion does **not** concentrate operational loss on
  any single class (0–4/40 nulls form a core) — itself a structural fact about code (callers of a
  class are scattered, not clustered). So Python's M1=0.96 is a raw recovery number; without a
  working null it is partly **mechanical** (the region is defined by operational loss, which M1
  measures). We flag this rather than over-claim a "+lift" for code.
- **The embedding detector M_emb is weak in Lean (0.63) and at chance in Python (0.50).** This is the
  *clean, null-independent* signal: text-embeddings are blind to graph-structural gaps, **more so in
  code than in Lean**. The article domain's *winning* detector is the *worst* available one here.

### The revision

Code does **not** sit between articles and Lean — on every available read (orphan-core degeneracy,
graph-region M1≫M_emb, M_emb-at-chance) it **patterns with Lean**. And the discriminating axis is
**not graph cleanness** (code's call graph is much noisier than Lean's proof DAG, yet the verdict is
the same). It is the **modality the gap lives in**:

> Articles' gaps are **semantic voids** — they live in embedding density, and a noisy citation graph
> can't see them, so the embedding wins. Lean's and code's gaps are **structural/usage gaps** — a
> concept embedded in the dependency structure but under-exercised — they live in the **graph**, and
> the text-embedding is blind to them, so the graph wins. *Cleanness was a confound; modality is the
> variable.*

This turns a predicted two-point *line through cleanness* into a *categorical* split by gap modality,
which is the more useful design rule: **pick the detector whose modality matches where the gap lives,
not by how clean the graph is.**

## Temporal result — git-tag holdout

Epoch-normalised git-tag holdout (each repo's release history mapped to E=6 shared epochs; node
"birth" = earliest epoch its qualified name appears; 92.7% of nodes resolved). At epoch T we rank
candidate concepts by `M1 = Δ` and ask whether **high-gap concepts attract more NEW operations**
(functions/tests born after T that call them) than a **growth-matched null** on (existing
operational-count, in-degree) deciles. Calibration first: a random ranking must give fill-ratio ≈ 1 /
AUC ≈ 0.5 — it does (AUC 0.49–0.52 across seeds/epochs), so the matcher is unbiased.

**Verdict: a clean NEGATIVE — gaps *anti*-predict arrivals (2 seeds, epochs e2/e3):**

| metric (citation attachment) | value | reading |
|---|---|---|
| M1 fill-ratio (top-gap vs matched null) | **0.04–0.18** | high-gap concepts attract *<20%* as many new callers as the null |
| M1 stratified AUC (centrality-matched) | **0.45–0.49** | at/below chance — no predictive signal above centrality |
| M1 raw AUC | **0.15–0.30** | *anti*-predictive — high-gap ⇒ *fewer* arrivals |

What actually grows is the **already-central** concept (`sklearn.base.BaseEstimator`,
`tornado.web.RequestHandler`, `PIL.ImageFile` all attract new operations), not the under-exercised
gap — **preferential attachment**, identical to the Lean (centrality-matched AUC 0.50) and article
(snapshot→arrival negative) temporal results. (Namespace-attachment is the one near-neutral signal,
M1 fill ≈ 0.8 / AUCstrat ≈ 0.53 — weak and not the primary citation oracle.)

So the temporal half is a **universal across every measurable substrate**: in articles, Lean, Python,
**Rust** (git-tag epochs; M1 fill-ratio 0.01–0.21, AUCraw 0.22–0.36 — anti-predictive), and **Go**
(`golang/go` release epochs, 94% name-resolution via a `go/parser` dumper; M1 fill-ratio 0.02–0.05,
AUCstrat ≈0.5; the saturated foundational interfaces `io.Reader`/`context.Context` are high-Δ yet
attract nothing new), a detected gap does not forecast the field's next unit — rich-get-richer does —
independent of substrate, language, or graph cleanness. (Bazel temporal is the one substrate that is
*not cleanly measurable*: historical `bazel query` fails under a modern pinned Bazel, so births
compress to recent epochs and the lib-vs-test target structure leaves no attributable arrivals — a
tooling limit, not a counter-result.)

## Stage 2 — Go: the CHA→RTA cleanness dial (the *controlled* test)

Stage 1 argued the axis is *modality, not cleanness* — but across articles/Lean/Python, cleanness and
modality are confounded (the domains differ in many ways at once). **Go removes the confound.**
`go/callgraph` ships **CHA** (over-approximate — an interface call points to *all* implementers) and
**RTA** (precise — reachability-pruned): two call graphs of *different cleanness on the SAME corpus*,
holding domain, embedder, granularity and scale fixed. If cleanness were the axis, the static winner
would shift CHA→RTA. **It doesn't.**

**Corpus:** the Go standard library (`-pkgs std`; dep-free, interface-rich, well-tested), extracted by
a `go/packages`+SSA+`go/callgraph` program (`adapters/go/extractor/`): **23,122 nodes** (3,617
types=B, 19,505 funcs=F); CHA **73,103** vs RTA **69,352** operational edges — the dial is real (e.g.
`net/http.socksConn` operational in-degree **88→1** as RTA prunes unreachable `io.Writer` dispatch).
Operational edges (func→type) come from **method-call resolution under each algorithm**, so the two
datasets differ *only* where dispatch precision differs.

**Result — the winner is invariant to cleanness** (plant-recover, graph region, 2 seeds each):

| Go std · region=graph | M1 (graph) plant AUC | M_emb (embedding) plant AUC |
|---|---|---|
| **CHA** (noisy) | 0.893 / 0.902 | 0.44 / 0.47 (chance) |
| **RTA** (clean) | 0.898 / 0.905 | 0.45 / 0.47 (chance) |

Cleaning the oracle (CHA→RTA) moves M1 by **<0.01** and leaves M_emb at chance. **The static winner
does not shift with cleanness** — the controlled confirmation of Stage 1: the gap's *modality* (it
lives in the graph) decides the detector, not how clean the graph is. The thesis's original
"static method ∝ cleanness" is refuted by its own sharpest test.

**Supporting:** the Lean embedding-orphan-core plant **fully degenerates on Go** (0/160 usable plants,
both resolutions — even more than Python's 2/160), and Go's citation↔embedding coupling is the
*weakest* of the three domains (**+0.037** cosine lift vs +0.11 in Lean/Python; top-10 caller fraction
0.05) — i.e. Go is the *closest to decoupled*, yet the graph detector still wins outright. Weak
coupling doesn't let the embedding find graph-gaps either.

**Caveat:** Go std is densely cohesive, so a random *local* function-deletion forms a graph-core about
as readily as a planted concept (local-null AUC 0.87–0.96) — M1's lift over that null is ≈0, i.e. M1
detects "operational support was removed," not concept-coherence specifically (as in Python; unlike
Lean's +0.13). The clean, null-independent signals are **M_emb-at-chance** and the **cleanness
invariance** across CHA/RTA. Temporal-Go not run (needs the Go repo with version tags; the installed
SDK is a single snapshot).

## Stage 3 — Rust: a 4th language (does the verdict generalize?)

Stage 2 settled cleanness; Rust asks whether the *modality* verdict (graph wins, embedding weak)
generalizes to a third code language with a very different type system (traits, generics, ownership).
Extracted with a **`tree-sitter-rust` parser in Python** (no Rust toolchain — source parsing only;
mirrors the Python `ast` adapter, deliberately noisy by-name resolution): 18 crates (`bitflags`…
`tokio`), **33,636 nodes** (3,802 types=B, 29,834 funcs=F), 49,446 edges. Top-Δ types are face-valid
(`tokio::AsyncWrite`, `clap::TypedValue` — traits implemented widely but constructed/exercised
narrowly).

**Result — Rust patterns with the others** (plant-recover, graph region, 2 seeds): M1 (graph) plant
AUC **0.961 / 0.918**, M_emb (embedding) **0.62 / 0.57**. The embedding-orphan-core plant
**degenerates** (0/160 plants) and coupling is **+0.057** (between Go's +0.037 and Lean/Python's
+0.11). Rust's M_emb is *slightly* above chance (its type-annotated signatures carry a little more
text signal than Python/Go's) but still loses decisively to the graph.

### The four-domain picture (Lean theorems + three code languages)

| domain | M1 graph plant AUC | M_emb embedding plant AUC | embedding-core plants | coupling |
|---|---|---|---|---|
| **Lean** (proof DAG) | 0.93 | 0.62 | 40/160 (forms) | +0.115 |
| **Python** (ast call graph) | 0.96 | 0.50 | 2/160 | +0.108 |
| **Go** (CHA / RTA) | 0.90 | 0.46 | 0/160 | +0.037 |
| **Rust** (tree-sitter) | 0.94 | 0.59 | 0/160 | +0.057 |

In **every** domain the **graph detector wins and the embedding detector is weak-to-chance**. The
verdict holds across a true dependency graph (Lean), three independent call/usage graphs
(Python/Go/Rust), the *controlled* CHA→RTA cleanness dial (Go), and a **3× range of
citation↔embedding coupling** (+0.037 → +0.115). Modality, not cleanness, decides the detector — and
it is robust to language, extraction method, graph cleanness, and coupling strength.

## Stage 4 — Monorepo build graph: the clean + coarse corner (the granularity cross)

The last substrate fills the one cell the others miss: a **clean** graph at **coarse** granularity.
Bazel build graphs are EXACT (`bazel query` reads the BUILD files — no static-resolution noise) but
**target-level** (a library, not a function), so this stage crosses cleanness with granularity. Nodes
= build targets (via `bazel query --output=xml`); **B = `cc_library`** (concepts), **F = `cc_test`/
`cc_binary`** (operations); edges = build `deps`. Corpus = abseil + protobuf + grpc — **931 targets**
(518 libraries, 413 tests/binaries), 4,564 edges. (At target granularity operational clusters are
small — median ~2 dependents — so the plant selector's cluster bounds were lowered, `--cluster-lo 3`.)

**Result — the verdict holds at coarse granularity** (plant-recover, graph region, 2 seeds): M1
(graph) plant AUC **0.86 / 0.87** with a genuine **positive lift +0.18 / +0.07** over the local null —
*cleaner* than the fine-grained code domains, because at target granularity the graph is sparse, so a
random local deletion rarely forms a core and the planted library genuinely stands out
(concept-coherent, like Lean). M_emb (embedding) **0.60 / 0.56** — loses decisively. Top-Δ libraries
are face-valid (`absl/base:nullability`, `absl/meta:type_traits`, `protobuf …compiler:code_generator`
— foundational utilities depended-on widely but exercised narrowly).

**The coupling twist:** Bazel's citation↔embedding coupling is the **highest** of all substrates
(**+0.161** — target labels like "absl strings str_split" are semantic *and* co-depending targets
share name tokens), yet the embedding **still** loses to the graph. So even when the embedding is at
its most informative, the graph wins for structural gaps — coupling *strength* is not the axis either.

### Final summary — six domains, one rule

| domain | gap modality | granularity | winning detector | losing detector | coupling |
|---|---|---|---|---|---|
| **Articles** (research) | **semantic-void** | work | **embedding** density-ratio (≈0.69) | graph-citation (noisy) | **+0.215** |
| **Lean** | structural | decl | **graph** citation-ratio (0.93) | embedding (0.62) | +0.115 |
| **Python** | structural | function | **graph** (0.96) | embedding (0.50) | +0.108 |
| **Go** | structural | function (CHA↔RTA *dial*) | **graph** (0.90) | embedding (0.46) | +0.037 |
| **Rust** | structural | function | **graph** (0.94) | embedding (0.59) | +0.057 |
| **Bazel** | structural | **target (coarse)** | **graph** (0.86) | embedding (0.59) | +0.161 |

**The detector is set by the gap's modality** — semantic-void → embedding, structural → graph — and is
invariant to everything else we varied: fine→coarse **granularity**, noisy→clean **edges** (the
controlled Go CHA↔RTA dial *and* Bazel's exact build deps), four **languages**, four **extractors**,
and — decisively — a **5.8× coupling range** (+0.037 → +0.215) in which the **single most
citation↔embedding-coupled domain, articles, is the one where the *embedding* wins.** Coupling
*strength* is therefore not the axis; modality is. In parallel, the **temporal** result is a universal
negative across every measurable domain (articles, Lean, Python, Rust, Go): a gap does not forecast
where the field's next unit lands — preferential attachment does. (`gap_recover/article_coupling.py`
computes the article coupling; `core/coupling_diagnostics.py` the code ones.)

## Follow-on experiments (2026-06-15)

### Modality fusion — strict substitutes (fusion HURTS)
*Backlog: `modality-fusion-and-predictor.md`.* Does fusing the two detectors beat either? Added a
z-scored `M1 + M_emb` fused detector (`core/plant_recover.py`) and ran the graph-region plant on all
five structural substrates (seed 12345):

| substrate | M1 (graph) | M_emb (embedding) | **M_fused** |
|---|---|---|---|
| Lean | 0.929 | 0.619 | 0.917 |
| Python | 0.961 | 0.458 | 0.960 |
| Go (rta) | 0.898 | 0.465 | **0.788** |
| Rust | 0.961 | 0.618 | 0.955 |
| Bazel | 0.859 | 0.599 | 0.800 |

**In every substrate fused ≤ M1, never exceeding it** — and where M_emb is at/below chance (Go, Bazel)
fusion *drags M1 down substantially* (Go 0.898→0.788). So for a given gap modality the off-modality
detector is not merely weaker, it is **anti-signal**: blending only dilutes. The modalities are
**strict substitutes** — the right move is to *pick* the modality-matched detector, not combine. (This
is equal-weight fusion; a learned weight would simply collapse to the graph detector, M_emb→0.) This
also largely answers the *a-priori-predictor* sub-question: modality is set by the gap **definition**
(known up front), not by an intrinsic substrate property — to be shown decisively by the within-corpus
conflict test (next).

### Change-form temporal — widening gaps don't get filled either (the lead fails)
*Backlog: `change-form-temporal-gaps.md`.* The one shot at a positive temporal signal: does ΔM1 (a gap
actively *widening* across epochs, `M1(T) − M1(prev)`) forecast arrivals where static M1 doesn't?
Compared static_M1 vs change_dM1 on the same candidates + growth-matched null (Python/Rust/Go births,
`change_form_temporal.py`). The **centrality-stratified AUC** is the de-confounded metric to read
here — the raw fill-ratio is uninformative at these sparse-attach sizes (its clustered-bootstrap
*random* baseline swings 0.09→14.7). Verdict:

- **change_dM1 is robustly *anti*-predictive in Python** (stratAUC ≈ **0.26**, consistent across 3
  epoch-pairs × 2 seeds), near-chance in Go (0.55), too-small in Rust (141 candidates).
- So the change-form lead **does not rescue prediction — it sharpens the negative**: concepts whose
  under-exercise is *accelerating* are the *least* likely to attract new operations (the gap
  compounds — poor-get-poorer). The temporal negative is robust even to differencing.

## Limitations / honest caveats

- **21/24 repos** in the static corpus: `sqlalchemy`/`matplotlib` use a `lib/` package layout and
  `poetry` a layout the discovery missed (logged, not silent). A 1-line `discover_package` fix would
  add them; unlikely to change the qualitative result.
- **Python null degeneracy** under `--region graph` means M1's code recovery has no working
  discriminating null; the load-bearing static claim rests on **M_emb-at-chance** + the **coupling**
  and **orphan-core-degeneracy** measurements, not on M1's raw AUC.
- The embedding arm uses bge-small on terse signatures; a code-specific embedder
  (`jina-embeddings-v2-base-code`) is the obvious robustness check (untried) — but note the embedding
  *coupling* is already strong, so a better embedder is unlikely to rescue *graph*-gap detection.
- Temporal `build_births` falls back to the wrong package dir for a few repos at their oldest tags
  (e.g. numpy→`weave`, scipy→`Lib`), shifting some early births later; `networkx`/`sqlalchemy` have
  no semver tags and are temporal-excluded.
- **Rust** uses a tree-sitter by-name resolver (no type info): `x.method()` dispatch is dropped and
  cross-crate refs are unresolved, so the graph is noisy/low-recall — *intended* (Stage 2 showed noise
  doesn't change the verdict), but the recovery has the same graph-region/null caveats as Python/Go.
  Temporal-Rust not run. As with Go/Python, the load-bearing signal is **M_emb-below-M1** + the
  coupling/degeneracy measurements, not M1's raw AUC.
- **Bazel** is small-scale (931 targets, 3 repos) with a lowered `--cluster-lo 3` (target-level
  operational clusters are small) and a terse-label embedding arm; `grpc //:all` returned only
  libraries (its tests live elsewhere). It is a single clean+coarse point, not a sweep — but it *does*
  carry a working positive M1 lift, unlike the dense fine-grained code domains. Its **temporal arm is
  the one substrate not cleanly measurable** — historical `bazel query` fails under a modern pinned
  Bazel, so births compress to recent epochs and leave no attributable arrivals (a tooling limit).

## Reproduce

```bash
conda run -p .conda python -m artifacts_scripts.curriculum.gap_recover.adapters.python.fetch_corpus
conda run -p .conda python -m artifacts_scripts.curriculum.gap_recover.adapters.python.build_data --out cache/code_gap/python/data
conda run -p .conda python -m artifacts_scripts.curriculum.gap_recover.adapters.python.build_births --data cache/code_gap/python/data
# static (graph region, the runnable code variant):
conda run -p .conda python -m artifacts_scripts.curriculum.gap_recover.core.plant_recover \
    --data cache/code_gap/python/data --region graph --min-region 1 --seed 12345 --out artifacts/code_gap/python/plant_graph_s12345.json
# temporal:
conda run -p .conda python -m artifacts_scripts.curriculum.gap_recover.core.temporal_holdout \
    --data cache/code_gap/python/data --seed 12345 --out artifacts/code_gap/python/temporal_s12345.json
```

*Generated by hand from the runs above; regenerate the JSON via the commands rather than editing
numbers in place. Raw result JSON lives under `artifacts/code_gap/python/` (gitignored).*
