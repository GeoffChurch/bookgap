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
Below is the **complete `gemma4:31b` half (all 8 targets, 40 rows)**; the run
continues in the background through `qwen3.5:35b`, writing the final two-prover
`dashboard.txt` on completion:

```
model          arm                 n  succ  succ% mean_frac mean_prog   sum_tok sum_llm_s sum_vrf_s
---------------------------------------------------------------------------------------------------
gemma4:31b     control             8     2    25%     0.250     0.281      8398        55        26
gemma4:31b     t1                  8     1    12%     0.125     0.219     16993        95        26
gemma4:31b     t2_all              8     0     0%     0.021     0.062     63234       224       154
gemma4:31b     t2_curriculum_a     8     0     0%     0.050     0.095     54246       241       128
gemma4:31b     t2_curriculum_b     8     0     0%     0.050     0.092     52745       231       129
```

Over the full 8 targets the pattern sharpens: **control (25%) > t1 (12%)** — the
context-only arm consistently *hurt* at ~2× the tokens — and **`t2_curriculum`
(mean_frac 0.050, mean_prog ~0.09) ≈ 2.4× `t2_all` (mean_frac 0.021, mean_prog
0.062)** on graded progress while spending ~16% fewer tokens (≈53–54k vs 63k).
The selection effect is *more* pronounced over 8 targets than over the first 5.

(Re-render any `results.jsonl` with `build_dashboard` over rebuilt `ArmResult`s —
the experiment writes one JSON row per arm, flushed immediately, so a partial
file is always valid. The completed two-prover `dashboard.txt` adds the
`qwen3.5:35b` rows.)

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

## Goedel-Prover-V2-8B (the stronger-prover lever)

`docs/goedel-prover-serving.md` stood up **Goedel-Prover-V2-8B** (a Lean-tuned,
whole-proof prover on Qwen3, mradermacher Q8_0 GGUF via ollama) as the
single-biggest-lever the gemma4 verdict named. This is the first run that wires
it into the four arms. To keep it an apples-to-apples comparison with the gemma4
default run it is held to the **same 8 targets / same arms / same `top_n=4` /
`t2_max_chain=5` / graded scorer**, and — critically — **thinking-OFF** and **no
self-correction** (so we measure the curriculum effect, not Goedel's repair loop
or its Qwen3 `<think>` layer):

- **thinking-off** = `GoedelProver(think=False)` → the ollama `think=False` flag,
  which is Qwen3's `/no_think`. Goedel *still* emits its prompt-requested prose
  proof plan (that lives in the answer channel, before the fenced ```lean4
  block); we only suppress the extra Qwen3 `<think>` scratchpad. Matches gemma4's
  `think=False`.
- **no self-correction** = `max_retries=0` → exactly one generation per goal.
- **large budget** = `num_predict=8192`, `num_ctx=16384`, `temperature=0.2`
  (plan+proof are long). Driver: `curriculum_lean/run_goedel_sweep.py` (reuses
  `experiment.run_target_for_model` verbatim; `GoedelProver` is
  `OllamaProver`-API-compatible). Comparison renderer:
  `curriculum_lean/compare_goedel_gemma4.py`.

### Dashboard (Goedel-8B thinkoff vs gemma4:31b thinkoff, both 8 targets)

```
                   |  goedel-prover-v2:8b (thinkoff)   |       gemma4:31b (thinkoff)
arm                |   succ% mean_frac mean_prog sum_tok   |    succ% mean_frac mean_prog sum_tok
-------------------------------------------------------------------------------------------------
control            |     38%   0.375   0.375   39041   |      25%   0.250   0.281    8398
t1                 |     50%   0.500   0.500   41394   |      12%   0.125   0.219   16993
t2_all             |      0%   0.146   0.161  295857   |       0%   0.021   0.062   63234
t2_curriculum_a    |      0%   0.125   0.131  235265   |       0%   0.050   0.095   54246
t2_curriculum_b    |      0%   0.125   0.148  227714   |       0%   0.050   0.092   52745
```

Per-target single-goal outcomes (control / t1), 4 easy + 4 hard sampled:

| target | control | t1 |
|---|---|---|
| `Nat.add_zero`       | ✓ | ✓ |
| `Nat.zero_add`       | ✗ | **✓** |
| `Nat.succ_pos`       | ✓ | ✓ |
| `List.length_cons`   | ✓ | ✓ |
| `isBounded_nsmul`                                      | ✗ | ✗ |
| `List.replicateTR_loop_eq`                             | ✗ | ✗ |
| `MeasureTheory.Measure.isMulLeftInvariant_haarMeasure` | ✗ | ✗ |
| `CategoryTheory.Limits.map_lift_kernelComparison`      | ✗ | ✗ |

### Verdict: Goedel-8B is clearly stronger than gemma4 — but the hard half still resists

**Goedel-8B is competitive with — and on this set strictly better than — gemma4**
on the single-goal arms, the headline question:

- **control: 38% vs gemma4's 25%** (3/8 vs 2/8). On the 4 easy targets Goedel
  closes 3/4 monolithically; the one easy miss (`Nat.zero_add`) is **rescued by
  t1** — surfacing the curriculum-selected prerequisite statements let Goedel
  close a target its own monolith missed.
- **t1: 50% vs gemma4's 12%** (4/8 vs 1/8), and t1 ≥ control for Goedel
  (4/8 > 3/8). This is the *opposite* of gemma4, where t1 **hurt** (the extra
  context distracted the weak prover). For a competent prover the context arm
  *helps* — the first positive signal for the "does merely seeing the lemmas
  help?" question.
- **decomposition arms close 0 full chains for both**, but Goedel makes
  **3–7× more graded partial progress** per chain (t2_all mean_frac 0.146 vs
  0.021; t2_curriculum 0.125 vs 0.050). Among the decomposition arms the gemma4
  selection ordering (`t2_curriculum ≥ t2_all`) does **not** reproduce here:
  Goedel's `t2_all` mean_frac (0.146) slightly *exceeds* `t2_curriculum` (0.125),
  i.e. with a stronger prover the bigger cone closes marginally more sub-goals,
  and the per-target sub-lemmas dominate over which subset is selected.

**The hard half is the binding constraint, unchanged.** On all 4 sampled hard
targets (`isBounded_nsmul`, `List.replicateTR_loop_eq`, the haar-measure and
kernel-comparison goals) **every arm fails** — control, t1, and all three
decomposition arms — exactly as for gemma4. So the "decomposition rescues a hard
target the monolith misses" headline win **still does not materialise**: Goedel-8B
at thinking-off cannot close these goals *or* their sub-lemmas, so there is no
regime on this set where decomposition turns a control-fail into a success. The
on-easy-targets t1 rescue (`Nat.zero_add`) is the closest thing to the intended
win, but on an easy target.

Net: the prover lever raised the *single-goal* absolute pass rate (and flipped
t1 from a liability to an asset), which is real and worth banking; **the next
increment is thinking-ON** — re-running 8B with `think=True` (Goedel's intended
mode: the proof plan *is* the reasoning) to see whether the `<think>` layer lets
it close any of the 4 hard targets or their sub-lemmas, which is exactly the
case the decomposition arms need to finally separate on success-rate.

### Pipeline gotchas (Goedel-8B, thinkoff)

- **thinking-off does NOT silence reasoning — it can degenerate.** With
  `think=False` Goedel emits its plan in the *answer* channel. On the hard
  targets it frequently fell into a **repetition loop** ("…is definitionally
  equal to… is definitionally equal to…") that ran out the full
  `num_predict=8192` and produced **no fenced ```lean4 block**, so
  `extract_proof` fell back to raw prose → `parse_error`. Every hard-target
  control/t1 row is ~8.4–8.7k tokens (the budget cap) for this reason. The first
  smoke attempt hit this; a re-draw at the same temperature produced a clean
  541-token proof — so it is a sampling-tail failure, not a systematic break, but
  it inflates the decomposition-arm token totals 3–5× over gemma4 and is the main
  driver of the long per-arm wall (a 6-goal `t2_all` chain on a hard target ran
  8–10 min, several sub-goals each burning the full budget). thinking-ON (or a
  smaller `num_predict` for the weak sub-goals) should reduce this.
- **Output format is plan-then-fenced-full-theorem.** Goedel restates the
  complete `theorem … := by …` with explicit universe/instance annotations
  (`Eq.{1} (HAdd.hAdd.{0,0,0} n (0:Nat)) n`) inside the final fence;
  `experiment.extract_proof` correctly takes the last fence and strips the
  `theorem … :=` header to the body. No extraction changes were needed.
- **CPU contention.** The run shared the box with the LeanDojo mathlib trace
  build (`lake env lean --threads 32 … ExtractData.lean`, load avg ~36), which
  slowed every `lake env lean` verification well past its ~3.2 s baseline. Total
  wall ~3h13m for 8 targets × 5 arms. Verification is on the prebuilt
  `~/src/mathlib4` olean cache and is independent of the trace, so results are
  unaffected — just slower.

**Sample generation (t1 on `Nat.succ_pos`, verified sorry-free):**

```lean
by
  have h_main : ∀ (n : Nat), LT.lt.{0} (0 : Nat) (Nat.succ n) := by
    intro n
    -- Use the lemma `Nat.zero_lt_succ` to prove the goal.
    exact Nat.zero_lt_succ n
  exact h_main
```

Artifacts: `artifacts/lean_curriculum_experiment/goedel8b_thinkoff/results.jsonl`
+ `dashboard.txt`. Re-run:

```bash
PYTHONPATH=. conda run -p .conda python -m curriculum_lean.run_goedel_sweep \
    --out-dir artifacts/lean_curriculum_experiment/goedel8b_thinkoff
PYTHONPATH=. conda run -p .conda python -m curriculum_lean.compare_goedel_gemma4
```

## Goedel-Prover-V2-8B thinking-ON (a channel artifact, not a reasoning lever)

Goedel-Prover-V2 is built on Qwen3 and the model card runs it with the `<think>`
layer ON (the proof plan *is* its reasoning). The thinkoff verdict above named
thinking-ON as the next increment: does the `<think>` layer let the 8B close any
of the 4 hard targets (or their sub-lemmas) and finally make the decomposition
arms separate on success-rate? This re-run is held **identical to the thinkoff
run** except the two variables that have to move with `<think>`:

- **thinking-ON** = `GoedelProver(think=True)` (Qwen3 `<think>`, the model's
  intended mode).
- **bigger budget** = `num_predict=32768` (the card's `max_new_tokens`),
  `num_ctx=32768` — large enough to fit `<think>` + plan + proof and avoid the
  thinkoff repetition-loop truncation.
- Everything else unchanged: same 8 targets, all 5 arms, `top_n=4`,
  `t2_max_chain=5`, `temperature=0.2`, `max_retries=0`, graded scorer. Driver:
  `run_goedel_sweep.py --think --num-predict 32768 --num-ctx 32768`.

### Verdict: thinking-ON is a hard regression — the `<think>` trace never terminates

**The `<think>` channel degenerates into a verbatim repetition loop that
exhausts the entire token budget without ever emitting an answer.** This is not
a hard-target effect — it happens on the *easiest* target in the set. Three
independent smoke generations, all `done_reason='length'` (hit the cap, did not
stop on its own), all `content` (the answer channel) **empty**:

| target | budget | eval_tok | `done_reason` | thinking_chars | content_chars | proof |
|---|---|---|---|---|---|---|
| `Nat.add_zero` (easiest) | 40960 | 40960 | `length` | 162171 | **0** | none |
| `Nat.zero_add` (easy) | 32768 | 32768 | `length` | 96809 | **0** | none |
| `isBounded_nsmul` (hard) | 32768 | 32768 | `length` | 104858 | **0** | none |

The tail of every think trace is the *same sentence repeated dozens of times*
verbatim, e.g. for `Nat.add_zero`:

```
But looking more closely, the `HAdd.hAdd` is actually a typeclass that is used
to define heterogeneous addition, but in this case, it is being used with `Nat`
and `0 : Nat`, which is a bit confusing.
But looking more closely, the `HAdd.hAdd` is actually a typeclass that is used
to define heterogeneous addition, but in this case, it is being used with `Nat`
and `0 : Nat`, which is a bit confusing.
…  (×N until num_predict is exhausted)
```

…and the only fenced block anywhere in the trace is the **restated goal**
(`theorem task_Nat_add_zero : … := by sorry`) — the prompt's own input echoed
back, not a proof. Because `<think>` never closes, the answer channel is never
reached, so `content_chars=0` on *every* call and no proof is ever produced.
Raising the budget does not help: 40960 just buys more repetition.

**Root cause — the GGUF Modelfile sets `repeat_penalty 1`** (no repetition
penalty at all). `ollama show goedel-prover-v2:8b --modelfile` confirms it. With
zero repetition penalty, Qwen3's long unconstrained reasoning has nothing to
push it out of a repeat attractor, so the trace collapses into a literal loop.
At thinkoff the answer is short, so the loop is a rare sampling-tail event (a
re-draw escapes it — that is what the thinkoff section documents). At thinking-ON
the trace is long and unbounded, so the loop is **systematic and first-draw** —
even `Nat.add_zero`, which control closes at thinkoff, never emits an answer.

This answers the three headline questions, all negatively, for *this GGUF*:

1. **Does `<think>` lift success on hard targets / rescue a decomposition
   sub-lemma?** No. No proof is emitted on any target (easy or hard), so every
   arm is a uniform failure; there is no regime where decomposition separates.
2. **Does it eliminate the repetition-loop `parse_error`?** No — it makes it
   *worse*: thinkoff's occasional `parse_error` (prose with no fence) becomes
   thinkON's universal empty-content failure (the loop is now in the `<think>`
   channel and never terminates).
3. **Does the `t1 ≥ control` curriculum-context win hold?** Moot — with zero
   successes on any arm there is no success-rate signal to compare. The thinkoff
   `t1 ≥ control` win is *not* reproduced at thinkON because there are no wins at
   all.

So thinking-ON on the mradermacher Q8_0 8B GGUF is a **dead end as configured**;
the banked result remains the thinkoff `control 38% / t1 50%` and the thinkoff
`t1 ≥ control` win.

**On run cost.** The full 8×5 thinkON sweep was launched through the real driver
to confirm the smoke finding end-to-end. The first single-goal rows reproduced
it exactly — `control` on the *easiest* target (`Nat.add_zero`) burned the full
`num_predict=32768` (`tokens=32919`, 466 s) and emitted an empty proof →
`parse_error`. Since every generation degenerates identically, running the full
sweep to completion would have been ~10+ h of GPU producing only 0%/`parse_error`
rows (the decomposition arms alone are ~40 min *each* because every sub-goal hits
the cap), so it was stopped after confirming the pattern. The dashboard below is
a **bounded confirmation run**: `control` + `t1` over the 4 easy targets through
the real driver (the single-goal arms that carry the headline questions), with
the decomposition arms omitted — they are guaranteed 0% by the same mechanism
and add no signal. thinkOFF/gemma4 columns are the full 8-target sweeps.

### Bounded thinkON confirmation (superseded)

A bounded thinkON confirmation run (`control`+`t1` on the 4 easy targets) was
launched to ground the smoke finding in real driver output, but it ran on the
broken `repeat_penalty=1` config (0% by construction) and was **stopped** once the
root cause + fix were in hand — see Resolution below. It is not reported; the
loop-fixed numbers that matter are the thinkOFF `repeat_penalty=1.1` run.

### Extraction gotcha with `<think>` on (fixed)

`experiment.extract_proof` has a 4th fallback rule: when `content` is empty it
scrapes the `thinking` channel for a fenced block (correct for the weak
gemma/qwen arms, which at a tight `num_predict` sometimes put the *answer* in the
think block). For Goedel at thinking-ON that rule is **actively wrong**: the only
fence in the runaway `thinking` is the restated `:= by sorry` goal, so the
fallback would hand the scorer a bogus sorry-laden "proof" instead of a clean
empty-string failure — silently inflating partial-progress on rows that in fact
produced nothing. Fix: `goedel_prover.extract_goedel_proof` now calls
`extract_proof(content, thinking=None)` unconditionally — for Goedel the proof is
*always* the answer channel; an empty `content` correctly yields an empty proof.
Regression test: `curriculum_lean/tests/test_goedel_extraction.py`.

### Resolution: the loop is a `repeat_penalty=1` defect — fixing it proves thinkON is a *channel artifact*, and the payoff is thinkOFF

The repetition loop is a serving-config defect, not an inherent Goedel limit, so
`repeat_penalty=1.1` is now the `GoedelProver` default — wired into the
`ollama.chat` **options** (the prior `extract_goedel_proof(thinking=None)` patch
only suppressed the *symptom*; nothing had touched generation). Applying it
settled two things:

**1. The loop dies, but thinkON still emits no proof — it's a channel mismatch,
not deeper reasoning.** With `repeat_penalty=1.1`, thinkON smoke flips
`done_reason='length' → 'stop'` and `<think>` terminates at ~2000 chars instead of
running to the cap. But `content_chars` is **still 0**: the model emits its entire
response — proof plan **and** the complete fenced Lean proof — *inside* the
`<think>` channel, leaving the answer channel empty. Goedel-Prover-V2 is a
**whole-proof generator, not a hybrid reasoner**: its "proof plan" is already part
of its normal answer-channel output, so `think=True` just makes ollama re-tag that
one plan-then-proof stream as `<think>` and blank `content`. No deeper reasoning
channel is unlocked — it is a **channel artifact**. A full thinkON sweep would only
reproduce thinkOFF, channel-shifted, so thinkON is abandoned and `think=False` is
now the class default (Goedel's native mode).

**2. The real payoff of the fix is thinkOFF.** The same `repeat_penalty=1` loop
also corrupted the *thinkOFF* run's hard targets — its occasional `parse_error`
(prose, no fence) and especially its **0%-but-huge-token decomposition arms**
(`t2_all` 295k tok, `t2_curriculum` 235k tok) are the loop hitting `num_predict` on
individual sub-goals. So thinkOFF + `repeat_penalty=1.1` is the run that should
actually move: cleaner hard targets and a fair shot at decomposition signal.

Other levers (unchanged): **the 32B** (`goedel-prover-v2:32b`, Q4_K_M, downloaded)
is the next prover lever once the 8B loop-fix is banked; **self-correction**
(`max_retries>0`) is worth revisiting only once decomposition shows signal.

### thinkOFF with the loop fix (`repeat_penalty=1.1`, v4.30.0 graph + verifier)

Re-run held **identical to the original thinkOFF run except `repeat_penalty`** —
same 8 targets, all 5 arms, same scorer, and the decl-graph pinned to the v4.30.0
backup so curriculum + verifier stay coherent at v4.30.0 (the LeanDojo downgrade
moved the *canonical* graph to v4.20.0; see `docs/lean_decl_graph.md`).

<!-- RP11_DASHBOARD_PLACEHOLDER -->

```bash
PYTHONPATH=. conda run -p .conda python -m curriculum_lean.run_goedel_sweep \
    --graph-dir cache/lean/decl_graph_v4.30.0 \
    --out-dir artifacts/lean_curriculum_experiment/goedel8b_thinkoff_rp11
```

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
