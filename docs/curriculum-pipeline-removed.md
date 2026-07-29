# The `artifacts_scripts/curriculum/` pipeline was removed

**What happened.** Commit `0f2d0ad` (*refactor!: delete the legacy curriculum pipeline — replaced
by the curriculum repo's generate app*) deleted the whole tree: 107 files, −35,050 lines. Its
replacement is the sibling **`curriculum` repo** (`../curriculum`), consumed here as an editable
path dependency (`pyproject.toml`).

A second, smaller change compounds the staleness: `6518955` moved the environment from a local
`.conda` to `uv`, so any `.conda/bin/python -m …` incantation in an older doc is doubly wrong —
wrong interpreter *and* wrong module.

**Why so many docs point at it.** The curriculum work generated a lot of results docs, handoffs,
plans, and per-book eval examples, each recording the exact command that produced its numbers —
which is the right habit, and exactly what makes them look runnable now that the producers are
gone. Roughly 48 docs under `docs/` name a module in that tree.

## What is still true

**The findings are.** Nothing about the deletion invalidates a measurement. Results docs like
`curriculum-dendrogram-quality.md` still describe what was observed and why it mattered; treat
them as evidence, not as instructions.

**The commands are not.** Every `artifacts_scripts.curriculum.*` module is gone, and so are the
`*_results.json` outputs that lived beside them under
`artifacts_scripts/curriculum/dendro_eval/` — that directory held only regenerable run artifacts
(its own `.gitignore` said as much) and was cleared once its producers went.

## Recovering a specific module

History retains everything:

```sh
git show 0f2d0ad^:artifacts_scripts/curriculum/<module>.py
git show 0f2d0ad --stat            # the full list of what went
```

Read one to port an algorithm. Do **not** re-add it here — the pipeline's home is the `curriculum`
repo now, and reintroducing a parallel copy is precisely the split-brain the refactor removed.

## Reproducing an evaluation

Rebuild it against the `curriculum` repo's generate app. The old harnesses assumed a bookgap-local
module layout, a `.conda` interpreter, and joblib caches that no longer exist, so porting the
*method* is cheaper than resurrecting the *code*.

---

Docs that reference the removed tree carry a one-line pointer back to this file. Where a doc is a
dated handoff or plan, its commands are deliberately left verbatim: those are a log of what was
run, not a recipe to follow.
