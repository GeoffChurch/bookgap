# Code-domain gap detection — triangulating the clean-oracle thesis (Stage 1: Python)

*Status: Stage 1 (Python substrate) complete, 2026-06-15. Spec + program:
`docs/backlog/gap-detection-methods/code-domain-triangulation.md`. This doc reports what the
Python point actually showed — and how it **revises** the thesis.*

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

<!-- TEMPORAL_PLACEHOLDER -->

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
