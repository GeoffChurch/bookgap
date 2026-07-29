# Smoke test for the curriculum sibling-repo coupling

bookgap consumes `curriculum` as an editable path dependency (`../curriculum`,
`pyproject.toml:54` + `:85`). Nothing verifies that bookgap's call sites still
match curriculum's API, so drift is discovered only when something here breaks
at runtime — and `py_compile` cannot catch it, because every failure mode is a
runtime signature or import-resolution error.

## It has already bitten twice, in three days, via two different mechanisms

Both were found on 2026-07-28 while fixing an 11-failure test suite; both had
landed in curriculum within the preceding 72 hours.

1. **Removed keyword argument.** curriculum `c4bd916` (2026-07-25) deleted
   `weight_by_freq` from `extract_topic_keywords`, replacing it with Monroe's
   informative Dirichlet prior (`prior_alpha_0`). Three bookgap call sites
   still passed it and raised `TypeError`: `keyword_helpers.py`,
   `wiki/wiki_curriculum_demo.py`, `wiki/wiki_gap_spike.py`. Two of those were
   committed *that same day* without anyone noticing they were already broken.

2. **Top-level package name collision.** curriculum `229316b` (2026-07-27)
   added a top-level `scripts` package with an `__init__.py`. Python resolves a
   regular package ahead of any namespace portion *regardless of sys.path
   order*, so it shadowed bookgap's `__init__`-less `scripts/`. Three test
   files broke with a misleading `No module named
   'scripts.capture_gap_parity_golden'` — the parent resolved, the submodule
   did not. Fixed by making bookgap's `scripts/` a regular package
   (`scripts/__init__.py`) plus `pythonpath = ["."]`.

The second class is the nastier one: any new top-level package in curriculum
can silently capture a bookgap import, and the error names the submodule rather
than the collision.

## Proposal

A fast, no-I/O test in `tests/` that fails loudly on drift:

- **Signature check.** For each curriculum symbol bookgap calls, assert the
  call is bindable — `inspect.signature(fn).bind_partial(**kwargs_we_pass)`.
  Cheaper and more robust than importing and invoking. Current surface:
  `extract_topic_keywords` (3 sites), `curriculum.core.pagerank.compute_pagerank`
  (`lexrank_bgnorm/engine.py:42-43`).
- **Shadowing check.** Assert each top-level package bookgap imports by bare
  name (`scripts`, and anything else in that position) resolves to a path
  *inside this repo*, i.e. `Path(mod.__file__).is_relative_to(REPO_ROOT)`.
  That catches collision class 2 generically, without enumerating curriculum's
  packages.

Keep it in the default `tests/` run so it gates every commit.

## Alternatives considered

- *Pin curriculum to a revision.* Wrong shape: it is a path dependency
  precisely so the two evolve together, and pinning trades silent drift for
  silent staleness.
- *Rely on the full suite.* This is what happens today. It catches drift only
  where a test already exercises the path — `wiki_gap_spike.py` had no test and
  stayed broken.

## Related

Curriculum has no git remote yet (see CLAUDE.md), so there is no CI on that
side to run a reciprocal check. Until there is, the guard has to live here.
