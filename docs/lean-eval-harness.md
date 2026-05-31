# Lean-verified eval harness (Phase 3)

Phase 3 of `docs/backlog/curriculum-standalone-extraction-and-lean.md`: synthetic
**task generation** (the breakage modes) plus a **live Lean-verified graded
scorer** over the mathlib4 decl-dependency graph. Builds on the Phase-2
interface (`docs/lean-curriculum-sanity.md`): `LeanGraph`, `decompose_target`,
`prerequisite_cone`.

The deliverable is `curriculum_lean/{verify,tasks,score}.py` (+ a Lean
metaprogram `curriculum_lean/lean/ExtractSignatures.lean`). The validation gate —
**original mathlib proof verifies as success, bare `sorry` verifies as
fail/partial** — passes on every sampled target. This doc records the
verification approach that works, the gate result, the task modes, and the
Phase-4 interface.

## The verification approach that works (`verify.py`)

**Never edit mathlib in place** (that rebuilds every dependent). Instead, per
check:

1. write a self-contained scratch file: `import Mathlib` + a short preamble +
   the spliced candidate declaration;
2. run `lake env lean <scratch>` from `~/src/mathlib4` (the prebuilt `.olean`
   cache makes the whole-library import ~3.2 s);
3. parse stdout/stderr into a **graded** `VerificationResult`.

**Why `import Mathlib` (the whole library) rather than a minimal import set.** It
is the simplest *robust* choice: the target decl's module and all transitive
premises are unconditionally in scope, so we never compute (and never get wrong)
a per-decl minimal import. The cost is ~3.2 s and ~6.6 GB RAM per check against
the cache — acceptable, and reported as `wall_seconds`.

**The graded signal** (continuous, not pass/fail):

- `type_checks` — no `error` diagnostics (the binary per-goal signal);
- `sorry_free` — no `sorry`/`sorryAx`/`declaration uses 'sorry'` warning. A proof
  that compiles *with* `sorry` `type_checks` but is **not** `sorry_free`;
  `success := type_checks and sorry_free`;
- `n_open_goals` — on an `unsolved goals` failure, the remaining-goal count
  (turnstile `⊢` count in the reported proof state) — partial-progress signal;
- `status` (`VerifyStatus`) — `SUCCESS` / `SORRY` / `UNSOLVED_GOALS` /
  `TYPE_ERROR` / `PARSE_ERROR` / `TIMEOUT` / `OTHER_ERROR`.

**Robustness.** Each check runs under a per-call `timeout` with
`start_new_session=True`, so the subprocess tree is reaped on expiry (never hangs
on adversarial agent output). Any toolchain failure is captured and classified
rather than raised. The classifier is unit-tested against captured real
`lake env lean` output (`tests/test_verify_parsing.py`), so it does not need the
toolchain to be tested.

### The crux that made the gate pass: re-parseable statements

The decl-graph `statement` column (the Phase-2 pretty-printed type) is **not**
reliably re-parseable, and the first gate run exposed exactly why. Three targets
(`ZMod.wilsons_lemma`, `ZMod.val_zero`, `FiniteField.prod_univ_units_id_eq_neg_one`)
failed `@<fqn>` verification with `type_error` because the default pretty-printer
**elides information that does not round-trip**:

- **coercion target types** — `↑(p-1)! = -1` lost the codomain, so on reparse
  Lean inferred `ℕ` and `-1 : ℕ` raised `failed to synthesize Neg ℕ`;
- **numeric-literal types** — `ZMod.val 0 = 0` left `0`'s type ambiguous
  (`OfNat (ZMod ?m) 0` stuck);
- **universe levels** — `∏ x, x = -1` over `Type u_1` reparsed with a fresh
  universe metavariable `?u` that would not unify with the original's `Eq.{1}`.

The fix is to re-derive each goal's type from the **loaded environment** (not the
graph string) with three pretty-printer options that force a round-trippable
form, via the `#sig` metaprogram (`lean/ExtractSignatures.lean`):

```lean
set_option pp.coercions.types true   -- ↑x : Cod
set_option pp.numericTypes  true     -- (0 : ZMod n)
set_option pp.universes      true     -- Eq.{1}, Type u_1, Eq.{u_1 + 1}
```

`#sig Foo.bar` then emits one tab-delimited line
`@@SIG@@<TAB>name<TAB>kind<TAB>one-line-type` (rendered at format width 100000 and
newline-collapsed so it stays on a single line). `extract_signatures(names)`
drives the metaprogram once for a whole batch, amortising the ~3 s import. With
these options, `theorem t : <type> := @<fqn>` type-checks sorry-free for every
sampled target — the harness's import/splice machinery is correct.

The scratch preamble also sets `set_option linter.unusedVariables false` (the
round-trippable types carry explicit binder names that a `@<fqn>` reference
leaves unused — a benign warning we silence to keep output clean).

## Ground truth = `@<fully.qualified.name>`

The canonical ground-truth proof of a decl's own statement is a **reference to
the original constant**, `@<fqn>`. It type-checks sorry-free by construction (it
*is* the mathlib proof term), which is exactly what validates the harness. The
human source proof is fragile to extract (namespaces, `open`/`section`,
equation-compiler `|` syntax) and is **not** required for harness validity; we
store `@<fqn>` as the ground truth.

## Validation gate result (the make-or-break check)

On a handful of generated `sorry` tasks (the Phase-2 sanity targets, one per
area), scored with both the ground-truth provider and a bare-`sorry` provider:

| Target | (a) ground-truth `@<fqn>` | (b) bare `sorry` | gate |
| :-- | :-- | :-- | :-- |
| `Finset.prod_Ico_id_eq_factorial` | success (3.18 s) | sorry, not-success (3.18 s) | PASS |
| `Subgroup.card_subgroup_dvd_card` | success (3.17 s) | sorry, not-success (3.19 s) | PASS |
| `ZMod.wilsons_lemma` | success (3.18 s) | sorry, not-success (3.19 s) | PASS |
| `IsPreconnected.intermediate_value` | success (3.16 s) | sorry, not-success (3.21 s) | PASS |

- **delete-cone** (`ZMod.wilsons_lemma`, top_n=4, 5-goal chain): ground-truth
  provider reconstructs the **whole chain** (`fraction_closed=1.00`); sorry
  provider closes **none** (`fraction_closed=0.00`, 5 sorries). Chain:
  `ZMod.val_zero → ZMod.nontrivial → FiniteField.prod_univ_units_id_eq_neg_one →
  Finset.prod_Ico_id_eq_factorial → ZMod.wilsons_lemma`.
- **derail** (`verify_broken=True`): picks `over_apply` (`@foo → foo _ _`), which
  genuinely fails (`parse_error`); the ground-truth fix verifies (success,
  edit-distance 5).

**Timing.** ~3.2 s per verification (dominated by `import Mathlib`); the
multi-goal delete-cone task was ~16 s for 5 goals. The full integration suite
(10 Lean-invoking tests) runs in ~110 s.

The gate passing on all four targets — including the three that *failed* before
the round-trippable-pp fix — is the evidence the splice machinery is right. The
first failing run (and its diagnosis) is preserved above precisely because it is
what proves the pp-option fix was necessary, not cosmetic.

## Task modes (`tasks.py`)

A `Task` carries `goals: List[Goal]` (each with `name`, a collision-free
`fresh_name`, the re-parseable `type_str`, and `ground_truth_proof = @<name>`),
plus mode-specific metadata.

- **`make_sorry_task(graph, target)`** — statement of `target` with its proof
  replaced by `sorry`; the agent re-proves. Single goal; ground truth `@<target>`
  (validates the harness).
- **`make_delete_cone_task(graph, target, top_n=…, cone_kinds=("theorem","def"))`**
  — take `target` + a connected prerequisite sub-DAG via `prerequisite_cone`
  (kind-filtered to drop typeclass-projection scaffolding), ordered bottom-up by
  `decompose_target`, with `target` appended as the final (frontier) goal. The
  agent reconstructs the chain in order; graded score = fraction reconstructed.
- **`make_derail_task(graph, target, perturbation=None, verify_broken=False)`** —
  perturb the ground-truth reference into a broken candidate (`over_apply`,
  `ref_misapply`, `naive_tactic`, …); the agent repairs it. `verify_broken=True`
  tries perturbations in order and keeps the first that actually fails (a stray
  tactic can accidentally close a trivial goal). Scored by edit-distance-to-fix +
  type-checks.

`sample_theorem_targets(graph, n=…, min_out_degree=2)` picks deterministic
theorem-kind targets with real premises (so the breakage is non-trivial).

## Graded scorer (`score.py`)

A **candidate provider** is any `Goal -> proof_string` callable (the agent, or a
baseline). `score_task(task, provider)` dispatches by mode, splices
`theorem <fresh> : <type> := <proof>`, verifies, and aggregates a continuous
`TaskScore`:

- `fraction_closed` — goals closed (sorry-free type-check) / total;
- `n_sorries_remaining`;
- `mean_partial_progress` — per-goal continuous credit
  (`success=1.0 > sorry=0.5 > one-goal-left=0.25 > … > hard-error=0.0`);
- `edit_distance_to_fix` (derail);
- `total_wall_seconds`; and placeholders the Phase-4 arms fill
  (`attempts`/`tokens`/`wall_clock_seconds`).

`cross_task_variance(scores)` aggregates a batch into a `BatchScore` with
`mean_fraction_closed` and `var_fraction_closed` (reliability across tasks).

Two baselines double as the gate: `ground_truth_provider` (`@<fqn>` → must
succeed) and `sorry_provider` (`sorry` → must fail/partial).

## Interface notes for Phase 4 (control / T1 / T2 arms)

An arm is a **candidate provider** `provider(goal: Goal) -> str` and runs through
the same scorer. The signatures the arms call:

```python
from curriculum_lean import (
    load_lean_graph, make_sorry_task, make_delete_cone_task, make_derail_task,
    score_task, cross_task_variance, ground_truth_provider, sorry_provider,
)

g = load_lean_graph()                          # Phase-2 LeanGraph (~9 s, ~100 MB)
task = make_delete_cone_task(g, target, top_n=8)   # or make_sorry_task(g, target)

# An arm = a Goal -> proof-string callable. `goal.type_str` is the re-parseable
# statement the agent must prove; `goal.fresh_name` is where it is spliced;
# `goal.name` is the original decl (for citing / context).
def control_arm(goal) -> str:
    return my_agent.prove(goal.type_str)       # no extra context

def t2_curriculum_arm(goal) -> str:
    # decompose_target(g, goal.name, top_n=k).ordered_names() gives the selected
    # prerequisites to surface in context (statements via extract_signatures).
    return my_agent.prove(goal.type_str, context=selected_prereq_statements)

score = score_task(task, control_arm)          # continuous TaskScore
# score.fraction_closed / .mean_partial_progress / .total_wall_seconds
# score.attempts / .tokens / .wall_clock_seconds  <- the arm sets these
batch = cross_task_variance([score_task(t, arm) for t in tasks])
```

- The provider returns **only the proof body** (`"by …"` or a term); the scorer
  builds the full `theorem … :=` and verifies it.
- For the T2 "see your own earlier proofs" ablation, splice the agent's earlier
  proofs as preceding declarations (via a richer candidate / `task.extra_opens`)
  so goal *k* is checked against the agent's reconstruction rather than the
  mathlib originals; the scoring loop is unchanged.
- Efficiency fields (`attempts`/`tokens`/`wall_clock_seconds`) are placeholders
  on `TaskScore` for the arm to populate; the harness records its own verify
  `wall_seconds` per goal in `goal.wall_seconds`.
- `verify_candidate(candidate, timeout=…)` and `verify_theorem(name, type, proof)`
  are the low-level entry points if an arm wants to verify intermediate attempts
  itself (e.g. a repair loop) before submitting.

## Interactive prover (Phase 4.5): tactic-by-tactic via LeanDojo `Dojo`

The one-shot providers above hand a whole proof to `lake env lean` once: the
model emits a complete `by …` block (or a term) and it either type-checks or it
does not, with **no feedback**. That is the Phase-4 binding constraint — a weak
local model routinely picks the *right lemma* but the *wrong Lean-4 surface
syntax* and dies one-shot.

`curriculum_lean/interactive.py` replaces that with a **tactic-by-tactic loop**
over LeanDojo's traced `Dojo` environment. Per step:

1. build a `TacticPrompt` from the current `state.pp` + the accepted-so-far
   tactic script + the arm's context;
2. ask the pluggable `TacticBackend` for the **next single tactic**;
3. `dojo.run_tac(state, tactic)` →
   - `ProofFinished` → QED (success);
   - new `TacticState` → accept the tactic, advance, reset the per-step repair
     counter;
   - `LeanError` → feed `result.error` back into the prompt and let the model
     fix *that one tactic*, up to `max_retries_per_step` times, then give up.

**Per-step error feedback is the repair loop** — built into the protocol, not
bolted on. The loop is bounded by `max_steps` (proof depth),
`max_retries_per_step` (repair), and `wall_timeout` (total), and records
attempts / tokens / wall-clock just like the one-shot path. The result's
`stop_reason` (`qed` | `max_steps` | `wall_timeout` | `step_exhausted` |
`given_up` | `empty_tactic` | `dojo_error`) classifies every exit.

The loop is written against **duck-typed** result objects (classified by
`_classify_result`: `.error`→error, class-name `Finished`→QED, `GivenUp`→given-up,
else a state) so it unit-tests with a `FakeDojo` and runs against the real
`lean_dojo` types interchangeably — importing `interactive.py` never requires a
trace, and `lean_dojo` / `ollama` are imported lazily only when actually proving.

**LLM backend is pluggable.** `TacticBackend.propose_tactic(prompt) -> TacticGen`
is the seam. `OllamaTacticBackend` wraps the current local models (gemma4 /
qwen3.5); the Goedel prover (or any API model) later only implements that one
method.

### How it slots into the four-arm ladder

The arms (`control` / `t1` / `t2_all` / `t2_curriculum`) decide *which goals* and
*what context*; interactive vs one-shot is just *how each goal is proved*. So the
interactive backend is a **drop-in candidate provider** `Goal -> proof_string`
(the same Phase-3/-4 contract): it opens a `Dojo` on the goal's theorem, drives
the tactic loop, and returns the assembled `by\n  <tac1>\n  <tac2>…` string,
which the **unchanged** graded scorer verifies exactly as a one-shot proof.

`experiment.py` exposes this via `run_experiment(..., interactive=True,
interactive_repo=<LeanGitRepo>)` (CLI: `--interactive`). It swaps
`OllamaProver.make_provider` for `_interactive_single_goal_provider` (control,
t1) and routes each `_score_decomposition` subgoal through
`prove_goal_interactive` (t2_all, t2_curriculum a/b) — **same** arm context
builders, **same** scorer. The one-shot path is untouched (default).

```bash
conda run -p .conda python -m curriculum_lean.experiment --mode pilot \
    --interactive --mathlib-commit c5ea00351
```

### LeanDojo 4.20.0 ⨯ Lean v4.30.0: two compat patches (the real blocker)

LeanDojo 4.20.0's Python is fine on v4.30.0, but **its two bundled Lean
metaprograms do not work against Lean v4.30.0** — both needed surgical patches
(originals saved as `curriculum_lean/lean/{ExtractData,Lean4Repl}.orig.lean`).
The single root cause is the v4.30.0 **String API overhaul**:

- `String.Pos` became a *dependent* type `(s : String) → Type`; the byte
  position is now `String.Pos.Raw` (with `.byteIdx : Nat`), and
  `Syntax.getPos?` returns `Option String.Pos.Raw`.
- `Substring` was renamed `Substring.Raw`.
- `String.trim` / `String.drop` now return a `String.Slice` (not `String`).

**`ExtractData.lean`** (the tracer — runs at `lake env lean --run
ExtractData.lean`): without the patch, `abbrev TraceM := StateT Trace MetaM`
fails to elaborate because `Trace`'s `pos : String.Pos` fields no longer name a
type, which cascades into every `TraceM Unit` mis-leveling (`Type 1` vs `Type 2`
errors) and the whole trace **fails at the extraction step**. Patch: emit
`String.Pos.Raw.byteIdx` / `Substring.Raw.toString` (the JSON byte-offset
contract that `LeanFile.convert_pos(int)` consumes is unchanged) and replace
`.trim`/`.drop` with `.trimAscii.toString`/`.drop ….toString`.

**`Lean4Repl.lean`** (the interactive REPL — built per-trace as a `lean_lib`):
this compiled with only deprecation warnings (so the trace "succeeds"), but every
`run_tac` crashed with `DojoCrashError: Unexpected EOF` even though `Dojo` init
returned the initial state correctly. Two distinct v4.30.0 changes:
- the three `.trim` calls → `.trimAscii.toString` (same `String.Slice` issue:
  feeding a `String.Slice` to `Json.parse` fails with "unexpected end of input");
- **the real fix — the stdin read.** Under `lake env lean <file>`, Lean
  v4.30.0's elaboration frontend **buffers/consumes process stdin**, so a
  tactic-level `IO.getStdin.getLine` (which is how the `repl` tactic reads each
  request) returns the **empty string** — the REPL parses `""` as JSON, throws
  `[fatal] failed to parse JSON offset 0`, and the subprocess dies (the EOF the
  Python side sees). Reading the **raw `/dev/stdin` handle**
  (`IO.FS.Handle.mk "/dev/stdin" .read`, opened once before the request loop)
  bypasses the frontend buffer and reads the request correctly. This was isolated
  with a one-line tactic probe: `IO.getStdin.getLine` → `len=0` inside a tactic,
  `/dev/stdin` → the full line. (Linux-specific; fine for this host/cluster.)

These patches are to the **installed** `lean_dojo` package files (the only copy;
LeanDojo hardcodes their paths and copies them into each traced repo at trace
time). They are exactly the fix the in-progress mathlib v4.30.0 trace needs at
*its* extraction step — i.e. without the `ExtractData.lean` patch the mathlib
trace would also have failed. **On any `pip install`/reinstall of `lean-dojo`,
re-apply both patches** (diff against the `.orig.lean` backups). The clean
long-term fix is upstreaming v4.30.0 support to LeanDojo.

### Toy-trace validation (no mathlib needed)

`curriculum_lean/lean/toy_repo/` is a tiny Lean-4-**core** repo (no mathlib:
`lean-toolchain` v4.30.0, `lakefile.toml`, `Toy/Basic.lean` with four trivial
tactic-mode theorems). LeanDojo traces it in seconds (`build_deps=False`). The
integration suite copies it into a temp git repo, traces it into an isolated
`CACHE_DIR` (never touching `~/.cache/lean_dojo`'s mathlib trace), opens real
`Dojo`s, and exercises the full loop:

- a GOOD tactic advances the state / reaches `ProofFinished` (QED detected);
- `constructor` splits one goal into two (multi-goal state handling);
- a BAD tactic surfaces a real `LeanError` whose text is fed back, and the
  loop's repair path recovers when the next proposal is the fix;
- `prove_interactive` closes a toy goal end-to-end with a scripted backend.

Loop logic (state advance, QED, per-step error-feedback/repair, the
step/retry/wall-clock budgets, empty-tactic & given-up paths, accounting) is also
unit-tested against a `FakeDojo` with **no toolchain** (default suite).

```bash
# loop logic (fast, no Lean):
conda run -p .conda python -m pytest curriculum_lean/tests/test_interactive_loop.py -q
# end-to-end on the toy trace (needs ~/.elan; traces in ~1-2 min):
conda run -p .conda python -m pytest \
    curriculum_lean/tests/test_interactive_toy_trace.py -m integration -v -s
```

### What remains to run on mathlib

Only the **completed** trace (`leandojo_trace_1780237584`, building 2026-05-31)
plus opening `Dojo`s on the sampled targets — no code change. `open_dojo_for_goal`
maps `goal.module` (`Mathlib.Data.Nat.Basic`) → `Mathlib/Data/Nat/Basic.lean`
and `goal.name` → the fully-qualified theorem; `prove_goal_interactive` returns
a `dojo_error` result (not a raise) if a target is not locatable, so one bad
target never sinks a sweep. Run with `--interactive --mathlib-commit c5ea00351`
once the trace cache is present.

## Files

- `curriculum_lean/verify.py` — scratch-file splice + `lake env lean` + graded
  `VerificationResult` (`verify_candidate`, `verify_theorem`, `VerifyStatus`).
- `curriculum_lean/interactive.py` — the interactive `Dojo` loop
  (`prove_interactive`, `InteractiveResult`), the pluggable `TacticBackend` /
  `OllamaTacticBackend`, `open_dojo_for_goal`, and the drop-in candidate provider
  (`make_interactive_provider` / `prove_goal_interactive`).
- `curriculum_lean/lean/toy_repo/` — tiny traceable Lean-4-core repo for
  validating the loop end-to-end without mathlib.
- `curriculum_lean/lean/{ExtractData,Lean4Repl}.orig.lean` — pristine backups of
  the patched LeanDojo metaprograms (re-apply the v4.30.0 patches after any
  `lean-dojo` reinstall).
- `curriculum_lean/lean/ExtractSignatures.lean` — `#sig` metaprogram (the
  round-trippable signature extractor).
- `curriculum_lean/tasks.py` — `extract_signatures`, the three breakage modes,
  `sample_theorem_targets`.
- `curriculum_lean/score.py` — `score_task`, `TaskScore`/`GoalScore`/`BatchScore`,
  `cross_task_variance`, the ground-truth / sorry baselines.
- `curriculum_lean/tests/` — `test_verify_parsing.py` + `test_tasks_structure.py`
  + `test_interactive_loop.py` (pure-Python, default suite) and
  `test_lean_harness.py` + `test_interactive_toy_trace.py` (Lean-invoking, marked
  `integration`, incl. the validation gate + the toy-trace interactive loop).

Run the gate:

```bash
conda run -p .conda python -m pytest curriculum_lean/tests/test_lean_harness.py \
    -m integration -v
```
