# Gap-driven test generation, scored by real coverage — code's clean oracle + the application

**Status: not started.** The static code plant-recover never got a clean ground-truth oracle (the
project's most-prized thing — Lean has its verifier, articles have none; the code graph-region M1 is
partly mechanical). **Code-coverage tooling IS that oracle**, and the application closes the code
generation loop (the article side already closed its placement loop, `docs/curriculum-close-loop.md`).

## The test
Pick a Python package (from the existing corpus). Detect its top-Δ concepts — classes that are
**structurally central but under-tested** (high structural-users, low operational/test-users). Have an
LLM generate tests targeting those concepts (use the **local ollama** adapter, NOT OpenAI — project
policy, memory `project_llm_coverage_ollama` / `docs`). Measure the **real coverage delta**
(`coverage.py` before/after) and compare:

- **gap-seeded** test generation (target detected high-Δ concepts), vs
- a **centrality-matched / random** baseline of equal generation budget.

**Question:** does seeding test generation at detected gaps raise measured coverage more than the
baseline? If yes, the gap signal is *actionable* and we have a real tool (auto-test-gen for
under-tested-but-central code), validated against an **external oracle** rather than a circular plant.

## Reuses / difficulty
`gap_recover` detection + the Python corpus + the ollama generation adapter (and `coverage.py`).
**Difficulty: hardest of the follow-ons** — LLM generation + coverage instrumentation + actually
running the target package's test suite in an isolated env. Watch capability-gating (memory: relevant
context lifts strong models ~6× but weak models barely benefit) — pick a capable local model.
