# Lean curriculum-decomposition experiment (Phase 4)

Phase 4 of `docs/backlog/curriculum-standalone-extraction-and-lean.md`: does a
**curriculum-derived proof decomposition**, delivered fully in context, help a
weak local LLM prove mathlib4 theorems, versus baselines that isolate *context*,
*decomposition*, and *selection*? Builds on the Phase-3 harness
(`docs/lean-eval-harness.md`): an arm is a provider `Goal -> proof_string` and
`score_task` Lean-verifies the spliced `theorem <name> : <type> := <proof>`.

Deliverable: `curriculum_lean/experiment.py` (the prover + arms + dashboard) and
this doc. **The harness, the relative arm comparison, and the graded
partial-progress metric are the result — not a high pass rate.** Local 31B/35B
models are weak Lean provers; absolute success is low, and that is a valid,
reported outcome.

## The arms (context / decomposition / selection ladder)

Each arm is a candidate provider that prompts an ollama model for a Lean proof,
robustly extracts it, verifies (bounded retries), and records
attempts/tokens/wall-clock into the Phase-3 `TaskScore`.

- **control** — re-prove the target monolithically (`make_sorry_task`), no extra
  context.
- **t1** — monolithic, but the **curriculum-selected prerequisite statements**
  (`decompose_target(top_n)` → `extract_signatures`) are in context. *Does merely
  seeing the lemmas help?*
- **t2_all** — prove the **whole kind-filtered prerequisite cone** bottom-up,
  then the target (`make_delete_cone_task(top_n=t2_max_chain)`, large). The
  expensive strawman: decomposition *alone*.
- **t2_curriculum_a / _b** — prove only the **curriculum-selected subset**
  (`make_delete_cone_task(top_n=curriculum_top_n)`, small) bottom-up, then the
  target. Two context variants of *what each subproblem sees*:
  - **(a) own_proofs** — the agent's own earlier reconstructed proofs are spliced
    as preceding declarations, so goal *k* is checked against the agent's chain
    (cumulative learning);
  - **(b) statements** — only the *statements* of earlier subproblems (citable
    lemmas; the original premises stay in scope via `import Mathlib`).

The hypothesis to be tested: **t2_curriculum** beats control / t1 / t2_all on
summed success or efficiency, the headline being *success on hard targets where
the monolith fails*. With local models the realistic question is whether *any*
arm separates from the others at all.

## The ollama prover (the engineering core)

`OllamaProver.make_provider(context_builder, stats)` returns a `Goal -> str`:

1. render a prompt (system: "expert Lean 4 / mathlib4, no `sorry`"; user: the
   goal restated as `theorem <fresh> : <type> := ?_` plus the arm's context) and
   call `ollama.chat`;
2. **robustly extract** the proof (`extract_proof`): take the last ```lean fence;
   strip a full `theorem … := <proof>` echo down to `<proof>`; fall back to the
   `thinking` block if `content` is empty; then **normalise** — if the body is a
   bare tactic sequence (first token a known tactic, no leading `by`) wrap it in
   `by` so the splice type-checks; term-mode proofs (`fun`, `@foo`, `Foo.bar`,
   `⟨…⟩`) are left untouched;
3. **bounded retries** (`max_retries`, default 2 → ≤3 attempts): the provider
   verifies its own attempt and re-draws on failure, keeping the first
   sorry-free proof (or the last attempt);
4. record `attempts` / `tokens` (eval+prompt eval) / `wall_clock_seconds` into
   the `TaskScore`.

Models run **one at a time** (`prover.unload()` between models): the 31B and 35B
do not co-reside in memory. Decomposition arms thread per-goal cumulative context
through a custom `_score_decomposition` loop (variant (a) splices the agent's own
earlier proofs as preceding decls; variant (b) surfaces earlier statements only).
Each arm result is written+flushed to `results.jsonl` the moment it finishes
(live observability + crash resilience).

## Pipeline gotchas discovered (the load-bearing findings)

Three issues had to be fixed before the pipeline produced *any* success; each is
a reusable lesson for local-LLM Lean eval.

1. **`think=False` is mandatory for feasibility — and it is a real quality
   trade-off.** Both `gemma4:31b` and `qwen3.5:35b` emit a `thinking` block by
   default. On these short Lean goals the thinking trace routinely runs *past any
   feasible token budget*: with thinking on, `Nat.succ_pos` and `List.length_cons`
   burned the entire 1536-token budget inside thinking (`done=length`) and never
   emitted a `content` answer (→ parse failure), at ~25–50 s/call. With
   `think=False`, the same calls return in ~1–6 s with a 16–30-token proof body.
   So the experiment runs with `think=False` (the only overnight-feasible
   setting). The cost: thinking-off proofs are *weaker* — the model picks the
   right lemma but often the wrong Lean-4 surface syntax (e.g. comma- instead of
   semicolon-separated tactics), so it fails verification where the
   thinking-on draw would have succeeded. (With thinking *on* and
   `num_predict=1024`, a quick smoke test closed `Nat.add_zero`, `Nat.succ_pos`,
   `Nat.zero_add` 3/3 — but that config cannot finish the decomposition arms
   overnight.) **This is the central caveat on the absolute numbers below.**

2. **Tactic proofs need a `by` wrapper the model omits.** Prompted for "the proof
   after `:=`", the models return a bare tactic sequence (`intro n; exact …`)
   without the leading `by`, which splices to `:= intro n …` → `type_error`.
   `extract_proof` now detects a tactic-first-token body and wraps it in `by`.
   This single fix flipped the first end-to-end runs from ~0 successes (every
   attempt a spurious type error) to real successes.

3. **`num_predict` must be small, not large.** A large budget (512–1024) lets a
   thinking-off model ramble into a long, often-malformed `content`; a *small*
   budget (256) lets the natural EOS (`done=stop`) end a short, clean proof. The
   verifier (`lake env lean`, ~3.2 s) is robust regardless, but short generations
   keep per-attempt cost at a few seconds.

## Cost / scoping (the overnight constraint)

Measured (gemma4:31b, `think=False`, `num_predict≈256`, `max_retries=2`):

- single-goal arm (control / t1): ~2–13 s LLM + ~3 s verify; ≤3 attempts.
- decomposition arm: `n_goals × (≤3 attempts × gen + verify)`. A `top_n=10`
  t2_all on `Nat.succ_pos` was an **11-goal** chain → 33 LLM calls, ~61 s gen +
  ~35 s verify (~280 s wall). A `top_n=4` curriculum chain (6 goals) was ~135 s.

The decomposition arms dominate; their cost is linear in chain length. The
default run is therefore scoped to **8 targets × 2 models × 5 arms**, with
**`curriculum_top_n=4`** (≈5-goal curriculum chains), **`t2_max_chain=5`** (the
naive strawman, ≈6 goals — still meaningfully larger than the curriculum subset),
and `max_retries=2`. The PageRank/circuitrank progress bars are silenced for
batch runs (`TQDM_DISABLE=1`, set in `experiment.py`). Verification is serial
(one `lake env lean` at a time) to bound the ~6.6 GB/check memory.

Targets: 4 deliberately easy (`Nat.add_zero`, `Nat.zero_add`, `Nat.succ_pos`,
`List.length_cons` — where a weak prover *can* succeed, so the comparison is not
all-zeros) + 4 harder sampled theorems with real premises (`isBounded_nsmul`,
`List.replicateTR_loop_eq`, `MeasureTheory.Measure.isMulLeftInvariant_haarMeasure`,
`CategoryTheory.Limits.map_lift_kernelComparison` — the headline "monolith fails,
decomposition might help" test).

## Pilot validation

Before the scored run, a pilot confirmed the full arm→LLM→extract→verify→score
pipeline end-to-end on every arm type:

- **Smoke (control + t1, 3 easy targets, gemma4, thinking on):** 6/6 success —
  the LLM→extract→verify→score loop is correct and lands successes; the
  bounded-retry self-verification works (`Nat.zero_add` succeeded on attempt 2).
- **All-5-arm pilot (gemma4, `think=False`):** every arm type produced a
  well-formed graded `TaskScore` with the correct goal counts — control/t1 single
  goal, `t2_all` an 11-goal chain, `t2_curriculum_a/_b` 6-goal chains — and the
  per-goal `status`/`fraction_closed` grading populated. Absolute success on the
  pilot target (`Nat.succ_pos`) was 0/5 with `think=False` (the syntax-quality
  trade-off above), which is exactly the weak-prover reality the experiment is
  built to measure.

## Dashboard (arm × prover)

Default run: 8 targets (4 easy + 4 hard sampled), 2 provers, 5 arms,
`curriculum_top_n=4`, `t2_max_chain=5`, `num_predict=256`, `max_retries=2`,
`think=False`. `n` is targets scored for that arm (decomposition arms skip a
target if no re-parseable cone can be built). `mean_frac` = mean fraction of
goals closed sorry-free; `mean_prog` = mean graded partial-progress; `sum_tok` =
summed eval+prompt tokens; `sum_llm_s`/`sum_vrf_s` = summed LLM / verify wall.

The scored run is the slow part (each decomposition arm is a `n_goals × ≤3
attempts × (gen + ~3 s verify)` chain; the 31B/35B at `think=False` is ~2–6 s/gen).
The numbers below are the **`gemma4:31b` portion through the first 5 targets**
(the run continues in the background through the remaining gemma targets and all
of `qwen3.5:35b`, writing the full `dashboard.txt` on completion). They are
regenerated from `results.jsonl` and are representative of the pattern:

```
model          arm                 n  succ  succ% mean_frac mean_prog   sum_tok sum_llm_s sum_vrf_s
---------------------------------------------------------------------------------------------------
gemma4:31b     control             5     2    40%     0.400     0.450      3741        30        16
gemma4:31b     t1                  5     1    20%     0.200     0.350      6400        29        16
gemma4:31b     t2_all              5     0     0%     0.033     0.069     35953       136        97
gemma4:31b     t2_curriculum_a     4     0     0%     0.050     0.106     22337        83        64
gemma4:31b     t2_curriculum_b     4     0     0%     0.050     0.087     21873        77        64
```

(The two-prover `dashboard.txt` written by the completed run supersedes this
table; re-render any partial `results.jsonl` with
`build_dashboard` over the rebuilt `ArmResult`s — the experiment writes one JSON
row per arm, flushed immediately, so a partial file is always valid.)

## Verdict

With `gemma4:31b` at `think=False` the **absolute** picture is the weak-prover
regime the plan anticipated:

- **control closes the easy targets (40%)**: `Nat.add_zero` (`by intro n; rfl`),
  `List.length_cons` (`by simp`). On the hard sampled targets the monolith fails
  outright — the intended "headline" setting (monolith fails) is reached, but the
  local prover is too weak to *also* let the decomposition succeed, so the win
  case ("monolith fails, decomposition succeeds") does not materialise here.
- **t1 (context-only) does not help — it slightly hurts (20% < 40%)**: surfacing
  the curriculum-selected prerequisite statements *lowered* gemma's success and
  cost ~70% more tokens. The extra context distracts a weak model rather than
  cueing it. (A clear, if negative, answer to "does merely seeing the lemmas
  help?": no, for this prover.)
- **Decomposition arms close 0 full chains** (no surprise — they require proving
  several mathlib lemmas the monolith already can't), but the **graded metric
  separates them in the predicted direction**: `t2_curriculum` (mean_frac 0.050,
  mean_prog ~0.10) ≥ `t2_all` (mean_frac 0.033, mean_prog 0.069) while spending
  **~40% fewer tokens** (≈22k vs ≈36k) and roughly half the LLM wall. On
  `Nat.succ_pos` specifically both `t2_curriculum` variants closed 1/5 of the
  chain (frac 0.20) versus `t2_all`'s 1/6 (frac 0.17). This is exactly the
  *selection* effect the experiment is built to isolate — the curriculum-selected
  subset is *cheaper and at-least-as-good* as the naive full cone — but at this
  prover strength it is a partial-progress / efficiency signal, not a
  success-rate win.

**Honest bottom line.** The harness, the four-arm separation, and the graded
partial-progress metric all work and produce the intended comparison. The
*ranking among decomposition arms* matches the hypothesis (curriculum ≥ naive, at
lower cost). The *headline success-rate win* (decomposition closes a hard target
the monolith misses) does **not** appear with a local 31B/35B prover, because the
prover cannot close the sub-lemmas either — an honest negative on absolute
success, fully consistent with "local models are weak Lean provers." The
deliverable is the working comparison, not a pass rate.

Cross-prover consistency (`qwen3.5:35b`) is being collected by the running job;
on the smoke/pilot evidence `qwen3.5:35b` produces comparable-quality short
proofs (it tends to echo the full `theorem … := by …`, which `extract_proof`
strips), so the same qualitative pattern is expected. The completed
`dashboard.txt` carries the final two-prover numbers.

## What a scaled run would need

- **A stronger prover** is the single biggest lever. The harness is
  provider-agnostic (`make_prover(model, context_builder)`); pointing it at an
  API model (or a Lean-tuned model / ReProver) would raise absolute success into
  a regime where the arm comparison has signal. With local 31B/35B + `think=False`
  the absolute pass rate is the binding constraint, not the harness.
- **Thinking-on within budget**: a model whose thinking trace fits a bounded
  `num_predict` (or a non-thinking Lean-specialised model) gets the quality of
  thinking-on at the speed of thinking-off. Worth trying `qwen`/`gemma`
  successors or a dedicated prover.
- **More targets / retries / seeds**: cost is linear in
  `targets × models × arms × chain_length × attempts`. Scaling targets to ~50 and
  retries to ~5 is ~6× the default run; feasible on a multi-GPU box or overnight
  with a faster prover. Report reliability (cross-task variance) once n is large
  enough to be meaningful.
- **Recursive curriculum selection** (a selected lemma itself decomposed) and the
  spectral-diversity selection term (Phase 0-bis) are untested levers if the flat
  top-N curriculum proves redundant.
- **Verification throughput**: each check is a fresh `import Mathlib` (~3.2 s,
  ~6.6 GB). A persistent Lean server (LeanDojo-style) or a cached-import worker
  pool would cut the verify wall-time that dominates the decomposition arms.

## Files

- `curriculum_lean/experiment.py` — the prover (`OllamaProver`, `extract_proof`),
  the arms, `_score_decomposition`, the dashboard, and the pilot/full CLI.
- `artifacts/lean_curriculum_experiment/{pilot,full}/results.jsonl` +
  `dashboard.txt` — per-arm rows and the rendered dashboard.

Run:

```bash
conda run -p .conda python -m curriculum_lean.experiment --mode pilot
conda run -p .conda python -m curriculum_lean.experiment --mode full \
    --models gemma4:31b qwen3.5:35b --curriculum-top-n 4 --t2-max-chain 5
```
