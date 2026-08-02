# Repo architecture and cleanup — making bookgap openable without wincing

**Living doc.** Working notes at the bottom; tick items as they land. Written 2026-07-31
ahead of the open-source release. Every claim below was measured against the tree at
`52adab1`; the measuring commands are recorded so they can be re-run after each phase.

Scope: structure, layering, dead code, dependency hygiene. **Not** covered here: secrets,
PII, licensing — those are tracked separately.

---

## 0. The one-paragraph diagnosis

The paper's pipeline is **16,123 LOC across 60 modules** inside `bookgap/`. The repo is
**~136,000 LOC of Python**. So ~88% of what a reviewer sees on `git clone` is not the thing
the paper is about, and it is not signposted as such. The core package is fine — it is
buried. The two structural moves that fix the first impression are (a) evict the two whole
subtrees that have nothing to do with books (`curriculum_lean/`, `artifacts_scripts/gap_recover/`
+ friends), and (b) give the remaining research scripts a home that visibly says "research
scripts, not the pipeline". Everything else is small.

---

## 1. Map of what exists

Tracked files / Python LOC, from `git ls-files`:

| tree | files | .py | Python LOC | what it is |
| --- | ---: | ---: | ---: | --- |
| `artifacts_scripts/` | 243 | 210 | 77,508 | ~12 independent research threads + 66 loose root scripts |
| `bookgap/` | 97 | 96 | 20,666 | the installed package; 16,123 LOC of it is the paper's path |
| `curriculum_lean/` | 82 | 61 | 19,083 | Lean/mathlib4 proof-decomposition experiments |
| `tests/` | 97 | 93 | 18,159 | core suite (735 collected) |
| `docs/` | 207 | 0 | — | 61 backlog, 46 `superpowers/` (1.37 MB), 38 root, 28 curriculum_eval_examples, 19 handoffs, 15 dead_ends |
| `scripts/` | 4 | 3 | 285 | two one-off data-prep scripts + a shell installer |
| `scope_manager.py` (root) | 1 | 1 | 821 | Textual TUI + Typer CLI, shipped via `py-modules` |
| `paper/` | — | — | — | **nested git repo (`sbu-algorithms/nextbook`), gitignored. Do not touch.** |

### The paper's critical path

Transitive closure of `bookgap.batch_generator` = **60 bookgap modules, 16,123 LOC**. It is
exactly what CLAUDE.md's Architecture section describes: `batch_generator` → `configs/` →
`cloud` (+ `cloud_index`, `corpus`, `subcloud`, `index`, `embedder`, `denoisers`,
`cloud_denoiser`, `deduplicators`, `openalex_ingest`, `snapshot_ingest`) → `gaps` +
`llm/gap_algorithms/` → `llm/generators/` → `llm/artifacts/`.

**36 bookgap modules are outside that closure.** Most are legitimately side threads
(`wikipedia/`, `toc/`, `maximal_bookless`, `depletion_test`), a few are outright dead (§4).

### The reproducibility path (what a reviewer must be able to re-run)

`paper/README.md` names exactly four generators plus a survey one, and their whole import
closure inside `artifacts_scripts/` is **seven modules**:

```
artifacts_scripts/paper/{baseline_vs_pipeline_table, generate_latex_tables,
                         generate_proposal_appendix, per_topic_performance_scatter}
artifacts_scripts/survey/{generate_survey, verify_survey_alignment}
artifacts_scripts/common.py
```

That is the entire cross-repo contract. The other ~203 files in `artifacts_scripts/` are not
part of it.

### Side research threads (`artifacts_scripts/` subtrees)

| subtree | files | LOC | topic | external importers |
| --- | ---: | ---: | --- | --- |
| `gap_detection/` | 46 | 18,692 | code-gap triangulation (Lean/OA/wiki detectors) | its tests + 1 edge from `gap_recover` |
| `wiki/` | 14 | 7,951 | Wikipedia↔OpenAlex shared space | **none** |
| `gap_recover/` | 47 | 6,916 | gaps in *source code* (Go/Rust/Lean/Python/Bazel adapters) | its tests only |
| `survey/` | 17 | 3,831 | human evaluation, Qualtrics export | **none** (paper `\input`s its output) |
| `paper/` | 5 | 3,414 | LaTeX table generators | **none** (paper `\input`s its output) |
| `osm_spike/` | 6 | 2,105 | OpenStreetMap building footprints | **none** |
| `historical_gap_analysis/` | 9 | 1,638 | do published books fill gaps? | **none** |
| `denoiser_analysis/` | 5 | 1,624 | denoiser comparison | **none — and it does not import** (§4) |
| `lexrank_bgnorm/` | 9 | 1,518 | LexRank background normalisation | **none** |
| `llm_eval/` | 10 | 1,053 | AdalFlow prompt optimisation | its tests only |
| `funding_spike/` | 4 | 1,024 | NIH RePORTER funding data | **none** |
| `lean/` | 3 | 101 | mathlib4 decl-graph extraction | **none** |
| root loose scripts | 66 | 27,861 | 30 of them are `gap_*` probes (15,058 LOC) | 4 imported by paper/survey |

The measured shape is the important part: **every subtree is a leaf.** Nothing in the repo
depends on them except their own tests. They can be moved one at a time with no cascade.

---

## 2. What should move out

### 2a. `curriculum_lean/` → its own repo `~/src/curriculum_lean` — **biggest single win**

19,083 LOC, 82 files, 24 of its own test files, a `lean/` subdirectory of Lean 4 sources,
and eight `gen_job_*.sh` overnight-run scripts. It is a mathlib4 theorem-proving study.

Evidence it does not belong:
- It **imports nothing from `bookgap`** (`grep -rn "bookgap" curriculum_lean/` → zero hits).
- **Nothing imports it** — no reference anywhere in `bookgap/`, `tests/`,
  `artifacts_scripts/`, `pyproject.toml`, or `scope_manager.py`. The only mentions outside
  its own tree are three `docs/lean-*.md` writeups.
- It depends on the *sibling* `curriculum` package, not on this one.
- Its tests are not in `tests/`, are not in `testpaths`, and never run.
- `docs/backlog/curriculum-standalone-extraction-and-lean.md` already plans this split:
  "Migrates to `~/src/curriculum` (+ split to `~/src/curriculum_lean`) on extraction."

Move the tree, plus `artifacts_scripts/lean/` (3 files, the decl-graph extractor it feeds
from) and the `docs/lean-*.md` / `docs/goedel-prover-serving.md` writeups. Drop
`"curriculum_lean"` from `[tool.setuptools] packages`. Nothing else changes.

**Cheap. Zero risk to tests or the pipeline.** This alone removes 14% of the repo's Python
and one of the two "why is this here?" top-level directories.

### 2b. `artifacts_scripts/gap_recover/` (+ `gap_detection/`) → `~/src/gap_recover` or a `research/` subtree

`gap_recover/` is a well-built mini-project: `core/`, `adapters/{python,rust,go,lean,bazel}/`,
its own Go modules with `go.mod`/`go.sum`, a tree-sitter Rust extractor. It is about gaps in
**source code**, not literature. It is the reason `tree-sitter` and `tree-sitter-rust` are
required dependencies of the bookgap distribution.

`gap_detection/` (18,692 LOC — the single largest subtree in the repo) is the code-gap
triangulation study; `gap_recover/core/plant_recover.py` is the only thing outside it that
imports it.

Both are self-contained leaves with their tests already isolated in `tests/gap_recover/` and
`tests/gap_detection/`. Either extract to a sibling repo (they are a paper's worth of work in
their own right) or, if that is too much ceremony, move both under `research/` (§2d).

### 2c. Spikes with no future → delete or archive

- `artifacts_scripts/osm_spike/` (2,105 LOC) — OpenStreetMap building footprints. No
  importers, no tests, gitignored 140 MB cache. An exploratory analogy probe.
- `artifacts_scripts/funding_spike/` (1,024 LOC) — NIH RePORTER. Same.
- `artifacts_scripts/llm_eval/` (1,053 LOC, last touched 2026-01-08) — AdalFlow prompt
  optimisation. This is the sole importer of `adalflow`, which drags **`mlflow`, `pyvis`, and
  `graphviz`** into the required dependency set (all three otherwise unimported, §5b).
- `artifacts_scripts/denoiser_analysis/` (1,624 LOC, last touched 2025-12-16) — **does not
  import**: both entry modules do `from bookgap.cloud import load_works`, a symbol that no
  longer exists. Superseded by `gap_denoiser_bakeoff.py` +
  `docs/denoiser_bakeoff_results.md`.

Per the no-legacy rule: if the diagnosis is worth keeping, it goes to `docs/dead_ends/` as
prose; the code goes away. Do not keep a broken tree "in case".

### 2d. Everything else research-flavoured → `research/`, out of the installed package

After 2a–2c, `artifacts_scripts/` still holds ~66 loose root scripts (27,861 LOC, of which 30
`gap_*` probes = 15,058 LOC) plus `wiki/`, `historical_gap_analysis/`, `lexrank_bgnorm/`.
These are real, cited-in-docs research, and deleting them would lose the record. But they
should not be *inside the installed distribution*.

Proposed:

```
research/                       # NOT in [tool.setuptools] packages
  README.md                     # "these are exploratory; the pipeline is bookgap/"
  gap_probes/                   # the 30 gap_* root scripts
  benchmarks/                   # dedup/hnsw/faiss/usearch smoke tests + benches
  wiki/  historical_gap_analysis/  lexrank_bgnorm/
  _shared/                      # what is today artifacts_scripts/common.py, keyword_helpers.py, scope_loader.py
```

and the reproducibility path promoted out of the noise:

```
reproduce/                      # or keep as artifacts_scripts/paper + survey, but say so
  paper/    survey/
  README.md                     # the four-generator table, mirroring paper/README.md
```

The rename is mechanical (`git mv` + `sed` the `artifacts_scripts.` prefix) but it touches
`tests/artifacts_scripts/`, `tests/test_gap_*.py`, `bookgap/inplace_resume.py`, and the
provenance footers baked into already-generated `.tex` files under `paper/`. **This is the
one genuinely risky step** — see §6.

A cheaper 80% version, if the rename is too much: leave `artifacts_scripts/` where it is,
drop it from `[tool.setuptools] packages`, and add an `artifacts_scripts/README.md` that
opens with "none of this is the pipeline; the pipeline is `bookgap/`" plus the four-generator
reproducibility table. Most of the first-impression win for none of the risk.

### 2e. `docs/superpowers/` → out of `docs/`

46 files, **1.37 MB**, individual plan documents up to 95 KB. These are agent-session plans
and specs, not project documentation. They are the largest thing in `docs/` and the first
thing a `ls docs/` shows after `backlog/`. Move to `.superpowers/` (already exists, already
gitignored-adjacent) or a `docs/archive/`, or drop them — the durable conclusions are already
in `docs/*.md` and `docs/dead_ends/`.

---

## 3. What should stay, restructured

`bookgap/` is in decent shape. The problems are (i) 96 modules in a nearly flat namespace,
(ii) `configs/` sprawl, (iii) a handful of misfiled modules.

### Proposed layout

```
bookgap/
  __init__.py
  cli/
    batch_generator.py        # the Hydra entry point (today at package root)
    scope_manager.py          # today a root-level module shipped via py-modules
  config/                     # today configs/, flattened (see below)
    __init__.py               # re-exports every *Config
    experiment.py             # ExperimentConfig  (today configs/config.py)
    components.py             # Gap/LLM/Batch/Output/Cloud/Data/Embedder/Index/Filter configs
    store.py                  # today configs/config_store.py
  corpus/                     # data acquisition + storage
    openalex_ingest.py  openalex_scopes.py  snapshot_ingest.py  pyalex_client.py
    corpus.py  deduplicators.py
  space/                      # the embedding space
    cloud.py  subcloud.py  cloud_index.py  index.py
    embedder.py  tf_gep_embedder.py
    denoisers.py  cloud_denoiser.py  rsdc/
  gaps/                       # detection + scoring
    detect.py                 # today gaps.py
    algorithms/               # today llm/gap_algorithms/ — these are NOT llm-specific
      kball.py  voronoi.py  deflation_strategies.py
    metrics.py  size_metrics.py  statistics.py   # today gap_metrics/gap_size_metrics/gap_statistics
    types.py                  # today types.py + gap_types.py, merged
  llm/                        # proposal generation only
    protocols.py  prompt_record.py  prompt_render.py  identifiers.py
    data_structures.py  coverage_analysis.py
    generators/  artifacts/
  infra/
    jobs.py  scope_ops.py  util.py  plot_utils.py  visualization.py
  research/                   # side threads that are genuinely bookgap-flavoured
    wikipedia/  toc/  maximal_bookless.py  depletion_test.py  mp_ks_test.py
```

Reasoning:

- **`llm/gap_algorithms/` is the worst misfiling in the package.** K-ball and Voronoi are
  pure geometry over a `Cloud`; they know nothing about LLMs. `bookgap.types.CellType` lives
  at the package root and imports *up* into `bookgap.llm.*` to build its enum — which is why
  `types.py ↔ llm/gap_algorithms/kball_algorithm.py` shows as a mutual dependency in the
  import graph (resolved today only by a function-local import at `kball_algorithm.py:65`).
  Moving gap detection out of `llm/` dissolves that.
- **`configs/` is 8 subpackages, 6 of which hold one file and an empty `__init__.py`.** Two
  (`index/`, `util/`) have **no `__init__.py` at all**, so they are implicit namespace
  packages while their siblings are not — a newcomer cannot tell which convention is
  intended. There are **zero YAML files**; every config is a dataclass. Twenty files, 466
  LOC, could be three files. This is a 30-minute change with a mechanical import rewrite.
- **`util.py` (438 LOC, fan-in 35)** is three modules wearing one coat: process singletons
  (`LOGGER`/`TIMER`/`MEMORY`/`ARTIFACTS_DIR`/`CACHE_DIR`), the OpenAlex taxonomy enums
  (`Domain`/`Field`/`SubField`/`Topic`/`WorkType`), and the `Filter`/`FilterPair`/`Filters`
  data model — plus four loose numeric helpers. `Filter`/`FilterPair` in particular are core
  domain types, not utilities, and they are why `util.py` needs a `TYPE_CHECKING` import of
  `Cloud` (the leaf everything imports type-referencing the top of the stack). Split into
  `infra/runtime.py`, `corpus/taxonomy.py`, `corpus/filters.py`.
- **Entry points should be console scripts.** `uv run python bookgap/batch_generator.py`
  works only because CWD is on `sys.path`; `[project.scripts]` declares only `scope-manager`.
  Add `bookgap = "bookgap.cli.batch_generator:main"` and the README instruction becomes
  `uv run bookgap ...` — which also happens to work from any directory.

If the full move is too much at once, the **three sub-moves worth doing regardless** are:
gap algorithms out of `llm/`, `configs/` flattening, and `util.py` splitting. Each is
independently mechanical.

---

## 4. Dead and superseded code — name-and-delete list

Everything here has been checked for importers across the whole repo (AST import graph,
`/tmp/.../imports.json`), not just grepped.

**Hard-dead, does not even import:**

- `bookgap/data_sources.py` — **1,205 LOC, the second-largest module in the package.**
  `import bookgap.data_sources` raises
  `ImportError: cannot import name 'SubCloud' from 'bookgap.cloud'` — `SubCloud` moved to
  `bookgap/subcloud.py` and this file was never updated. Zero importers. Last touched
  2026-05-24 by a repo-wide `SCOPES_DIR` sed, i.e. nobody has *run* it in longer than that.
  It is still cited as live in `docs/cache_cleanup.md` and two backlog entries — fix those
  when deleting.
- `artifacts_scripts/denoiser_analysis/{denoiser_analysis,eigenbasis_analysis}.py` —
  `from bookgap.cloud import load_works`, symbol gone. (§2c)

**Empty files:**

- `bookgap/batch_utils/batch_operations.py` — 0 bytes, and `bookgap/batch_utils/` has no
  `__init__.py`. The whole directory is a ghost.
- `bookgap/llm/openai_generator.py` — 0 bytes.
- `bookgap/llm/json_prompt_generator.py` — 0 bytes.

**Duplicated definitions where one copy is unreferenced:**

- `bookgap/llm/gap_functions.py::DistsAndIdxs` duplicates
  `bookgap/gap_types.py::DistsAndIdxs`. `gap_functions.py` has zero importers → delete.
- `bookgap/llm/types.py::CellTypeRegistry` duplicates `bookgap/types.py::CellType` (same
  KBALL/VORONOI → class mapping). `llm/types.py` has zero importers → delete.

**Unimported, un-run:**

- `bookgap/path_management.py` — 516 LOC, zero importers, last touched 2025-09-17. Its
  docstring says "Moved from `config_manager.py` to avoid circular dependencies";
  `config_manager.py` no longer exists.
- `bookgap/inplace_resume.py` — 220 LOC, zero importers, and it is the **only** place
  `bookgap/` imports *up* into `artifacts_scripts` (`from artifacts_scripts.resume import
  build_run_slug, slugify` at line 17). Superseded by `batch_generator --resume-output-dir`,
  which the README documents. Deleting it also removes the layering violation.

**Stale test scaffolding — and this one has a dependency consequence:**

- `tests/config_manager/` contains **only** `conftest.py` + `__init__.py` and **no tests**.
  The conftest exists to skip `test_config_manager_step*` files that no longer exist, and it
  uses the removed `pytest_ignore_collect(path, config)` hookimpl signature — which is
  precisely why `pyproject.toml` carries `pytest==8.4.2` with a comment explaining that
  pytest 9 would turn it into a hard collection error. **Deleting this directory unpins
  pytest.** Textbook no-legacy-shim case: the pin exists to accommodate a conftest that
  guards tests that were deleted.
- `tests/test_results_openai_last_success/` — 3 committed OpenAI *output* files sitting in
  the test tree, not a test package.

**Broken/orphaned scripts:**

- `artifacts_scripts/output_plots.py` (1,544 LOC) does `from common import ...` (line 49) and
  `from generate_latex_tables import ...` (line 1335). The latter module has not existed at
  that path since it moved to `artifacts_scripts/paper/`, and bare `from common import` only
  resolves if `artifacts_scripts/` itself is on `sys.path` — it is not under
  `python -m artifacts_scripts.output_plots`, which is how `artifacts_scripts/README.md` says
  to run it. Documented, unrunnable.
- Same bare-sibling-import pattern in `artifacts_scripts/survey/plot_*.py`
  (`import verify_survey_alignment`) and `gap_recover/adapters/{python,rust}/extract.py`
  (`import util`).

**Stale references in docs (cheap, embarrassing if a reviewer follows them):**

- CLAUDE.md and TESTING.md both give a mypy command excluding `bookgap/batch_utils_OLD` —
  **that directory does not exist.**
- TESTING.md's last block references `bookgap/config_manager.py` and
  `bookgap/llm/llm_implementation.py`; neither exists.
- `docs/cache_cleanup.md` documents six `@MEMORY.cache` decisions in `bookgap/data_sources.py`,
  a file that cannot be imported.

---

## 5. Cross-cutting cleanups

### 5a. Layering

- `bookgap/` → `artifacts_scripts/`: exactly one edge, `inplace_resume.py:17`. Delete the
  file (§4) and the package becomes cleanly downstream-free. Then a lint rule
  (`ruff` `flake8-tidy-imports` banned-api, or just a test) can *keep* it that way.
- **No true runtime import cycles** inside `bookgap/` — checked with module-level,
  non-`TYPE_CHECKING` imports only. But there are **26 function-local `from bookgap.*`
  imports**, which is what a cycle looks like once it has been papered over. The concentrated
  ones are `cloud.py` (5), `snapshot_ingest.py` (4), `scope_ops.py` (4), and
  `kball_algorithm.py` (3, the `types.py` knot from §3). The `configs/` flattening and the
  gap-algorithm move together remove most of the reason for them.

### 5b. Dependency hygiene — high value for a *reproducibility* release

`pyproject.toml` declares 38 runtime dependencies. Measured against actual imports:

**Declared but imported nowhere in the repo** (checked with `/usr/bin/grep -rn` across
`bookgap artifacts_scripts curriculum_lean scripts tests scope_manager.py`):

`accelerate`, `bitsandbytes`, `graphviz`, `mlflow`, `pylatexenc`, `pyvis`, `umap-learn`.

Three of those (`mlflow`, `pyvis`, `graphviz`) are carried explicitly "for adalflow", whose
only importer is `artifacts_scripts/llm_eval/` (§2c). And `umap-learn` is the sole reason for
this pin:

```toml
"numba==0.63.1",
"llvmlite==0.46.0",   # pinned because umap-learn -> pynndescent resolves badly
```

Nothing imports `umap`. **Dropping `umap-learn` deletes both pins.** `bitsandbytes` alone
pulls a CUDA-linked wheel into every fresh `uv sync`.

**Imported but not declared** (currently satisfied only transitively, i.e. by luck of the
lock):

`scipy` (92 files), `scikit-learn` (40), `torch` (7), `requests` (9), `joblib` (6),
`pydantic` (2), `python-dotenv`, `rich`, `urllib3`, `botocore`, `usearch`, `lean-dojo`,
`aiohttp`, `pexpect`.

`joblib` is the `MEMORY` cache, `pydantic` is the entire `Proposal` schema, `scipy`/`sklearn`
are load-bearing in `denoisers.py` and `cloud.py`. For a repo whose stated reason for going
public is reproducibility, the declared dependency surface being both over- and under-stated
is the wrong first impression. Fixing it is a `pyproject.toml` edit plus one `uv sync`.

Once §2 lands, the remaining thread-specific deps (`tree-sitter*`, `adalflow`,
`fast_hdbscan`, `cdopt`, `coroica`, `pymanopt`, `pyriemann`, `boto3`, `seaborn`, `plotly`)
should be `[project.optional-dependencies]` groups, not baseline requirements. The pipeline
itself needs roughly: `hydra-core`, `omegaconf`, `numpy`, `scipy`, `scikit-learn`, `pandas`,
`pyarrow`, `faiss-cpu`, `sentence-transformers`, `torch`, `openai`, `pydantic`, `pyalex`,
`joblib`, `matplotlib`, `tqdm`, `psutil`, `python-dotenv`.

### 5c. Test organisation

735 tests collect in 7.3 s — the suite is healthy. The *layout* is not:

- Tests for `artifacts_scripts` code live in **two** places: `tests/artifacts_scripts/`,
  `tests/gap_detection/`, `tests/gap_recover/`, `tests/llm_eval/`, and six
  `tests/test_gap_*.py` at the tests root — *and* `artifacts_scripts/gap_detection/test_gap_detection.py`,
  `artifacts_scripts/lexrank_bgnorm/tests/`, `curriculum_lean/tests/` (24 files, never run).
  `pyproject.toml` even documents the split ("artifacts_scripts carries its own tests… run
  those by explicit path"), which is the tell. **One rule: tests mirror the tree they test.**
  When a subtree leaves (§2), its tests leave with it — that is most of this problem solved
  for free.
- `tests/` root has 22 loose files alongside seven themed subdirectories; the loose ones are
  a mix of core (`test_util`, `test_corpus`, `test_columnar_mask`) and research
  (`test_gap_deflation_bakeoff.py` — 99 KB, the largest test file in the repo).
- 4,507 LOC of `bookgap/` has no test importer, but **1,941 of that is code slated for
  deletion in §4** (`data_sources` 1,205 + `path_management` 516 + `inplace_resume` 220). The
  genuine gaps after cleanup are `llm/gap_algorithms/deflation_strategies.py` (443 LOC, a
  production code path), `gap_size_metrics.py` (272), and `gap_statistics.py` (251).

### 5d. Tooling and first-contact files

- **No CI.** For a repo whose release argument is reproducibility, a single GitHub Actions
  job running `uv sync --extra dev && uv run pytest tests/` is the highest-credibility-per-line
  file in the whole plan.
- **No lint config**, but `.ruff_cache/` exists — so ruff has been run ad hoc with defaults.
  Add a `[tool.ruff]` block so the setting is a decision rather than an accident.
- **No mypy config in `pyproject.toml`** — the invocation lives only in prose in CLAUDE.md
  and TESTING.md (and is stale, §4). Move it to `[tool.mypy]`.
- `.vscode/launch.json` and `.vscode/settings.json` are tracked and contain personal
  debug configurations for named sweep presets. Untrack.
- `artifacts/book_gaps/koehn_toc.json` is the single tracked file under an otherwise
  gitignored `artifacts/` tree — either it is a fixture (move to `tests/fixtures/`) or it is
  an accident.
- `README.md` currently advertises `curriculum_lean/` and "the legacy curriculum pipeline
  slated for replacement" in its Architecture section. After §2 it should describe: the
  pipeline, the four reproducibility generators, and where the research lives. That paragraph
  is what a reviewer actually reads.

---

## 6. Migration order

Ordered by (first-impression impact) ÷ (risk). Each phase leaves the repo green.

### Phase 0 — free wins, no code moves (½ day, zero risk)

1. Delete the §4 hard-dead list: `bookgap/data_sources.py`, `bookgap/path_management.py`,
   `bookgap/inplace_resume.py`, `bookgap/batch_utils/`, `bookgap/llm/openai_generator.py`,
   `bookgap/llm/json_prompt_generator.py`, `bookgap/llm/gap_functions.py`,
   `bookgap/llm/types.py`. **~2,000 LOC, all with zero importers, one of which does not even
   import.** Verify with `uv run pytest tests/` (should be unchanged at 735) and the mypy
   command.
2. Delete `tests/config_manager/` → then **remove the `pytest==8.4.2` pin** and re-lock.
3. Delete `tests/test_results_openai_last_success/`.
4. Fix the stale doc references (`batch_utils_OLD` in CLAUDE.md + TESTING.md,
   `config_manager.py`/`llm_implementation.py` in TESTING.md, `data_sources.py` in
   `docs/cache_cleanup.md`).
5. `pyproject.toml`: drop the seven unimported deps + the two numba/llvmlite pins; add the
   fourteen undeclared ones; `uv sync` and run the suite.
6. Untrack `.vscode/`.

*Breakage watch:* none of these are imported. The only real risk is (5) — an undeclared
transitive dep that the lock currently pins loosely. Do it as its own commit so a bisect is
one step.

### Phase 1 — evict `curriculum_lean/` (½ day, zero risk)

`git mv` the tree to `~/src/curriculum_lean` (with `artifacts_scripts/lean/` and the
`docs/lean-*.md` writeups), drop it from `[tool.setuptools] packages`, prune the `ollama`/
`lean-dojo`/`pexpect` deps it owned, update `README.md`'s Ecosystem section. Nothing in
bookgap references it, so `uv run pytest tests/` cannot notice.

**This is the single largest visible improvement per unit of risk in the whole document.**

### Phase 2 — signpost `artifacts_scripts/` (½ day, zero risk)

Do *not* rename yet. Instead:

1. Drop `"artifacts_scripts"` from `[tool.setuptools] packages` — it is a research tree, not
   a distributable. (Tests still import it: `pythonpath = ["."]` is already set in
   `[tool.pytest.ini_options]`, so collection is unaffected. **Verify this before
   committing** — it is the one non-obvious coupling in this phase.)
2. Rewrite `artifacts_scripts/README.md` to open with "none of this is the paper's pipeline"
   and carry the four-generator reproducibility table from `paper/README.md`.
3. Delete `osm_spike/`, `funding_spike/`, `llm_eval/`, `denoiser_analysis/` (§2c) — writing
   up whatever diagnosis is worth keeping into `docs/dead_ends/` first. Dropping `llm_eval/`
   lets `adalflow`+`mlflow`+`pyvis`+`graphviz` go.
4. Delete `artifacts_scripts/output_plots.py` (broken, §4) or fix its two imports.
5. Move `docs/superpowers/` out of `docs/`.

### Phase 3 — restructure inside `bookgap/` (1–2 days, contained risk)

In this order, each its own commit, each gated on `uv run pytest tests/` + mypy:

1. **`configs/` → `config/`**, flattened to three modules. Purely mechanical; ~20 import
   sites in `bookgap/`, ~40 in `artifacts_scripts/` and `tests/`. *Watch:*
   `hydra.main(config_path="configs")` in `batch_generator.py` and every
   `register_configs()` caller.
2. **Gap algorithms out of `llm/`** → `bookgap/gaps/algorithms/`. Merge `types.py` +
   `gap_types.py`. This is what dissolves the `types ↔ kball_algorithm` knot and lets three
   function-local imports become top-level.
3. **Split `util.py`** into `infra/runtime.py` (singletons), `corpus/taxonomy.py` (enums),
   `corpus/filters.py` (`Filter`/`FilterPair`/`Filters`). Highest fan-in in the repo (35
   modules) so the diff is wide, but every edit is a one-line import rewrite. Keep `util.py`
   as a re-export shim **only for the length of the commit** — then delete it in the same PR.
   (No shims survive the PR; that is the rule.)
4. **Subpackage the rest** per §3, and add `[project.scripts] bookgap = ...`.
5. `scope_manager.py` → `bookgap/cli/scope_manager.py`; drop `py-modules` from
   `[tool.setuptools]`; keep the `scope-manager` console script pointed at the new path.

*Breakage watch for the whole phase:* `bookgap.jobs` launches commands **as strings** through
`systemd-run`, and `cache/.jobs/*.json` manifests record those strings. Renaming a module
that appears in a launch command breaks *replaying* an old manifest, and `bookgap.scope_ops`
builds `uv run python -m bookgap...` command lines. Grep for `-m bookgap` and
`bookgap/batch_generator.py` in `jobs.py`, `scope_ops.py`, `scope_manager.py`,
`artifacts_scripts/do_sweep.py`, and every `*.sh` before renaming. Also: `@MEMORY.cache` keys
are joblib-derived from the *qualified function location*, so moving a cached function
(`gaps.py`, `cloud.py`, `openalex_scopes.py`) silently invalidates its cache — expensive, not
incorrect. Say so in the commit message.

### Phase 4 — extract the code-gap threads (1 day, moderate risk)

`gap_recover/` + `gap_detection/` to their own repo or `research/`. Their tests
(`tests/gap_recover/`, `tests/gap_detection/`) go with them. The single cross-edge
(`gap_recover/core/plant_recover.py` → `gap_detection`) means they move together.
Drops `tree-sitter`, `tree-sitter-rust`, and the vendored Go modules from this repo.

### Phase 5 — the `research/` rename (risky, optional)

Only if Phase 2's signposting proves insufficient. Renaming `artifacts_scripts` →
`research` rewrites a module prefix that is **baked into provenance footers of already-generated
`.tex` files in the `paper/` repo** (`% python -m artifacts_scripts.paper.generate_proposal_appendix …`)
and into `paper/Makefile`'s `make artifacts` target. That is a cross-repo, cross-commit
contract. If done, do it as a single atomic commit in both repos, and regenerate all four
`.tex` files so the footers agree.

---

## Status / working notes

- 2026-07-31 — measured and drafted. Nothing executed.
- Measurement artifacts (regenerate rather than trust): import graph built by AST-walking
  `git ls-files '*.py'`; reverse index by inversion. Reproduce with an AST walk that resolves
  each dotted name to the longest matching module path — the key outputs were
  (a) transitive closure of `bookgap.batch_generator`, (b) reverse-importer sets per module,
  (c) `from X import Y` where `Y` is not top-level in `X`.
- Open question: whether `gap_recover`/`gap_detection` (25 k LOC combined) deserve their own
  repo or just `research/`. `docs/code-gap-triangulation.md` suggests they are a paper in
  their own right, which argues for a repo.
- Open question: `artifacts_scripts/wiki/` (7,951 LOC) has an active backlog entry
  (`curriculum-generalization-pedagogical-subspace`) — it is live research, so it stays, but
  it may want to follow the curriculum engine to `~/src/curriculum` rather than sit in
  `research/`.
