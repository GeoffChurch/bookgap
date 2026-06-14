# PRE-REGISTRATION (v2, SIGNED) — curriculum-context effects on Lean theorem proving

*Status: SIGNED OFF by owner 2026-06-11 ~07:05 ("SGTM" on the filled draft); nothing here is to be revised after the confirmatory run
starts except by documented amendment. Lineage: pilot_v1 (frame v1, exploratory), matrix50 v2
(model×mode calibration + frame probes), the 2026-06-10 adversarial review (C1–C3, M1–M11) and its
fixes (audit, GT gate, provenance, append-mode, raw tee).*

## 1. Question

Does curriculum-derived context — prerequisite lemma statements (t1) or worked examples
(fewshot_curriculum) selected by dependency-graph centrality — improve a prover LLM's audited
success rate over (a) no context, (b) lexical retrieval at equal budget, (c) the same set shuffled,
(d) irrelevant context at equal count? Secondary: does the effect differ between a prover-tuned
specialist and a generalist (capability-relativity), and does the worked-pair unit beat the
statement unit?

## 2. Frame (target population)

- Source graph: mathlib v4.20.0 decl graph (357,486 nodes), trace-coherent (LeanDojo trace
  c211948581…); verifier `~/src/mathlib4_v4.20.0`; everything version-pinned.
- Family: **Mathlib.Probability.Kernel** — per the pre-committed band rule: matrix50 fresh-block 32B-clean per-target baselines Kernel 25% (in band) / Matroid 12.5% / Ordinal 0%; n=8/family caveat recorded; Ordinal retained as fallback if Kernel's full-frame gate attrition exceeds 15%.
- Targets: n = 140 sampled (≈130 expected post-gate), stratified 3×3 (traced-proof-size tertiles × intra-family
  cone-size tertiles), one-tactic-trivial theorems excluded as targets (eligible as context),
  ground-truth-gated (`#sig` round-trip + `@fqn` closes anonymized statement, sorry-free) with
  attrition reported. Pilot/matrix targets are EXCLUDED from the confirmatory set (run-id and name
  exclusion lists recorded in the run dir).

## 3. Arms (within-subject; every target sees every arm)

control · t1 (curriculum statements, top_n=4) · t1_shuffled (same set, order randomized) · bm25
(equal-K lexical retrieval) · irrelevant (equal-K module-disjoint placebo) ·
few-shot arms DROPPED per the entry criterion (cell 6: fewshot_fixed 0pp, fewshot_curriculum 0pp lift; both raised only circular closes). AMENDMENT (pre-start, documented): t1's context renderer is the FILTERED form (theorem-kind entries only, honest labels) — cell 6 showed the unfiltered renderer was a defect (instance/structure plumbing mislabeled 'theorem') that suppressed the effect under measurement (gemma: 0 clean targets unfiltered context vs 2 targets/0 circulars filtered); this is a bug fix to the shared renderer, applied to ALL statement arms (t1, t1_shuffled, bm25) identically.

## 4. Prover(s) and decode protocol

- Primary prover: Goedel-32B Q4_K_M (matrix: 22 clean rows/11 targets vs 8B's 5; quant caveat: 32B@Q4 vs 8B@Q8 — scale and quant confounded, noted in every table). Secondary (capability-relativity):
  gemma4:31b-it-qat thinkoff QUALIFIES (cell 6: t1_filtered lifted it 0→2 targets, 3/3 draws, 0 circulars) — included as the capability-relativity secondary.
- Decode: temperature 0.2; `num_predict` per-prover calibrated (Goedel 8192 — the 2048 cap
  demonstrably truncation-poisoned the pilot; gemma 2048; thinkon excluded per matrix cells 4–5 (gemma-on 0 clean/49 targets; 32B-on 5/5 parse — channel artifact confirmed on both Goedel sizes)); per-call seeds derived from (run_id, arm, target, draw); model digest +
  full options in every row; `num_ctx` pinned and recorded.
- Draws: 3 per (arm, target). Primary metric computed at pass@3.

## 5. Metrics (pre-committed)

- **Primary: audited pass@3 per arm** — success only if the Lean verifier accepts AND the
  constant-dependency audit confirms the proof does not reference the withheld target or its
  bare-segment alias (`success_circular` and `success_unaudited` are separate categories, never
  pooled with success).
- Secondary: circular-close rate per arm (name-recall measurement); tokens per solved target
  (total, including failed draws); graded partial-progress ONLY as a labeled exploratory table with
  a sensitivity analysis across credit weightings (review M5); waypoint-usage (did successful
  proofs cite surfaced context names — from contexts.json + proof terms).
- Analysis: per-target paired comparisons, McNemar (exact) for each arm vs control and t1 vs each
  control arm; report discordant-pair counts, not just rates. No interim peeking between draws;
  the analysis script is committed before the run.

## 6. Hypotheses and falsification

- H1 (curriculum-context): audited pass@3(t1) > control, AND > bm25 at equal K. *Falsified if t1 −
  bm25 ≤ 0 — then "curriculum" is just premise retrieval.*
- H2 (ordering): t1 > t1_shuffled. *Falsified if equal — selection, not ordering, carries the
  effect (still a result; the claim narrows).*
- H3 (content vs warmth): t1 > irrelevant. *Falsified if irrelevant ≈ t1 — context-warmth artifact.*
- H4 (capability-relativity, secondary): the t1−control delta is larger for the stronger prover.
- H5 (difficulty model, observational): target-level audited success is monotone-decreasing in
  premise surprisal (s_sum) — Spearman over the confirmatory rows, pre-committed direction.
  matrix50 prior: WITHIN-band prediction validated (similar-stratum 35% > anchors 22% per-target at 32B), CROSS-family absolute calibration refuted (fresh 'easiest' stratum 13%) — H5 is therefore committed as a within-family rank claim only
- Power note: n=137, within-subject; for control≈30%, detecting +15pp at α=.05, power: at control≈25% (Kernel probe) and target +15pp, McNemar power ≥0.8 needs ≈110–140 discordant-capable pairs — n=130 post-gate is adequate; if the realized full-frame control baseline falls below 15%, n extends to 200 BEFORE the run starts, never after.

## 7. Budget & runtime

from matrix pacing (32B@8192 ≈ 1.7–2 min/row): 5 arms × 130 × 3 draws ≈ 1,950 rows ≈ 55–65 GPU-hours for 32B + ≈8h for the gemma secondary (2 statement arms × 130 × 3 at ~20s/row) — a long weekend; tokens/verifier-interactions are the recorded cost metrics, wall-clock is not

## 8. What is already banked (and excluded from confirmation)

- pilot_v1 (frame v1): exploratory; established the truncation defect (fixed), the audit's
  necessity (gemma 0 clean / 6 circular), 32B ≫ 8B (final: 22 vs 5 clean rows; 11 vs 5 targets); gemma 0 clean in both think modes; 32B-thinkon channel artifact confirmed 5/5.
- matrix50 v2: calibration — model×mode matrix, frame-probe baselines, difficulty-model priors.
- All rows carry run_ids; the confirmatory analysis ingests only the confirmatory run_id.

## 9. Amendment rule

Any post-start change to arms, metrics, n, or decode protocol requires a dated amendment section
here stating what changed and why, before the changed rows are generated.

## 10. Pre-launch execution record (filled at launch, before first GPU row)

- **Sign-off**: owner "SGTM" 2026-06-11 ~07:05 on the fully-filled draft.
- **Frame built**: `artifacts/lean_frame_v3/kernel_targets_ungated.json` — n=140 sampled from
  1,013 traced Kernel theorems (granularity 2), 3×3 strata balanced 15–16/cell, seed 0; all 182
  previously-probed names excluded (8 were in-family — exactly the matrix50 fresh-block Kernel
  probes; exclusion lists recorded in the artifact's `_meta`). Builder: frame_selection.py
  machinery with family fixed by the band rule (NOT Borda-selected); script preserved at
  `/tmp/build_kernel_frame.py`.
- **Gate**: 125/140 pass = **10.7% attrition ≤ 15%** → Kernel CONFIRMED, Ordinal fallback not
  triggered (0 fail_sig, 15 fail_verify) → `artifacts/lean_frame_v3/kernel_targets_gated.json`.
  n=125 post-gate (vs ≈130 expected; power note in §6 unaffected — extension rule binds only if
  the realized control baseline falls below 15%).
- **Amendment applied** (the §3 filtered-renderer bug fix, pre-start): t1 → `t1_filtered_names`
  renderer; t1_shuffled → permutation of the SAME filtered set
  (`fewshot.make_t1_filtered_shuffled_context`); bm25 corpus restricted to theorem-kind
  (`controls.DEFAULT_CONTEXT_KINDS = ("theorem",)`; index 239,240 docs); `irrelevant` was already
  theorem-only (no change). Base t1 ranking untouched.
- **Validation**: 168/168 curriculum_lean tests pass post-amendment; 5-arm dry-run on gated
  Kernel targets shows honest theorem-only labels in every statement arm, t1 vs t1_shuffled
  identical sets in different orders; no new mypy errors in touched files.
- **Confirmatory run_ids**: `confirm_v3_goedel32b` (primary), `confirm_v3_gemma_off` (secondary)
  — the analysis ingests ONLY these.
- **Operational amendment (2026-06-11 ~08:25, at row 9 of 1,875; documented before any changed
  row generated)**: execution schedule changed from target-blocked (target→arm→draw) to
  DRAW-OUTERMOST (draw→target→arm), owner request. Statistically inert by construction — per-row
  seeds and anonymized fresh names derive from (run_id, arm, target, draw)/name identity, never
  from execution position, so the sampled completions and all §5 metrics are unchanged; the gain
  is that each completed draw layer is a fully-paired (target × arm) replicate (interruption
  robustness). A resume-skip was added at the same time (banked (arm, target, draw) rows of the
  same run_id are skipped on restart; the 9 pre-amendment rows are retained — identical seeds).
  The §5 no-interim-peeking rule is REAFFIRMED: layer boundaries create complete-looking pass@1
  datasets; no arm contrast is computed before the sweep ends. 168/168 tests pass post-change.
- **Owner reclassification (2026-06-11 ~12:30)**: "Forget about forking paths — this is just
  exploratory experimentation, not policy-making." The no-interim-peeking clause is demoted to
  a labeling convention: interim arm contrasts may be computed and discussed at any time and
  are labeled exploratory/interim; the instrument-rigor provisions (pairing, seeds, audit
  demotion, run_id isolation, append-only rows) remain in force; the final report still
  computes the §5 metrics on the full data.
- **Pause (2026-06-12 ~00:30)**: sweep paused at 337/1875 rows (draw 0: 326/625) to free the
  GPU for a datascience-3 book build (owner directive). Zero data loss — resume-skip replays
  nothing: relaunch `/tmp/run_confirm_v3.sh` and banked (arm, target, draw) rows of run_id
  confirm_v3_goedel32b are skipped. Watchdog + runner stopped via task kill (verified no
  survivors); model unloaded. Interim exploratory state at pause: t1_shuffled +9.2% vs control
  (complete-blocks n=65, McNemar p=.031); 1 tokens=0 row in 325 post-restart (base rate).
- **Execution-relocation amendment (2026-06-13 ~00:15; ADDITIVE, instrument-only — no change to
  arms/metrics/n/decode/seeds, run_id stays `confirm_v3_goedel32b`)**: to resume the paused sweep
  off the local GPU, proof GENERATION was to be distributed to the Stony Brook cluster (goedel-
  prover-v2:32b under per-job `ollama serve`, 6 balanced target shards) with VERIFICATION + the C1
  audit staying on-prem on the canonical Lean+Mathlib toolchain (`curriculum_lean.verify_deferred`,
  `~/src/mathlib4_v4.20.0` + `~/.elan/bin/lake`). Generation is single-shot (`--max-retries 0`,
  n_attempts=1) so it never consults the verifier; per-row seeds/fresh-names derive only from
  (run_id, arm, target, draw), so relocating generation is statistically inert and resume-skip on
  (run_id, arm, target, draw) replays nothing. Every cluster row carries `gen_host`. The verify
  relocation itself is proven exact (`--validate-banked` reproduced 75/75 banked verdicts on-prem).
  **STATUS: BLOCKED at the generation-side signature pre-extraction, NOT launched.** The cluster is
  Lean-free by design, but `cluster_gen` (verbatim copy of `run_real`'s sig-cache block) pre-seeds
  `sig_cache` only for the gated TARGETS and then `extract_signatures(need − targets)` the ~1000
  context-arm dependency statements — a `lake env lean #sig` call that `run_real` runs locally. On a
  Lean-free cluster node this raises `RuntimeError: lake not found on PATH`. The smoke gate (1 target
  × 5 arms × 1 draw, run_id `smoke_v3_goedel32b`) FAILED at this step: 0 gen rows, 0 proof_full,
  before any model generation. No fan-out was launched; no confirmatory `confirm_v3_goedel32b` rows
  were produced on the cluster; the 337 banked rows are untouched. Resolving the relocation requires
  supplying the context signatures Lean-free (a precomputed sig_cache the generator can load), which
  the current `cluster_gen` does not support without modifying pre-registered code — deferred.
- **Execution-relocation amendment — RESOLVED (2026-06-13 ~01:35; instrument-only, no change to
  arms/metrics/n/decode/seeds; run_id stays `confirm_v3_goedel32b`)**: the generation-side signature
  pre-extraction is resolved by giving the GENERATION HARNESS (`cluster_gen`, an untracked tool — NOT
  the pre-registered scientific code) a Lean-free `--sig-cache PATH` loader. A complete sig_cache was
  pre-extracted ON-PREM (which has Lean) reproducing `run_real`'s exact build: the 125 gated TARGET
  signatures pre-seeded from the gate report + the 1,000 CONTEXT-arm dependency signatures via the SAME
  `tasks.extract_signatures` (`#sig`) code path — **1,125/1,125 names resolved, all found-and-
  reparseable, 0 MISSING** (`artifacts/lean_frame_v3/sig_cache_v3.json`; the `need` set sourced from the
  banked `confirm_v3` `contexts.json` manifest, identical to recomputing it via `_selection_names`). On
  the cluster, `cluster_gen --sig-cache` LOADS this cache in place of the `lake #sig` call and ASSERTS
  `need − loaded == ∅` (FAILS LOUDLY, non-zero exit, on any missing/`#sig`-MISSING context name — never
  silently truncates a prompt, the failure mode `controls._render_decls` would otherwise hide). The
  byte-identity that preserves validity was VALIDATED: with the loaded cache, `cluster_gen`'s assembled
  prompt equals `GoedelProver`'s own first-attempt prompt (`assemble_prompt_for` vs the provider path,
  the `test_assembled_prompt_matches_provider_path` invariant) for **90/90 samples = 5 arms × 6 targets ×
  3 derived seeds, byte-identical**; every content arm renders non-empty context (t1/t1_shuffled 742
  chars, bm25 2053, irrelevant 1613; control empty by design) — confirming the cache wires context in
  with no `_render_decls` drop. **RE-SMOKE PASSED** (1 target × 5 arms × 1 draw, run_id
  `smoke_v3_goedel32b`, isolated from the confirmatory set; generated on cluster node ai04 then verified
  on-prem on the canonical Lean+Mathlib toolchain): `cluster_gen` skipped the `lake #sig` extraction
  (`need=12, all found`), produced non-empty `proof_full` for all 5 arms (sizes 3382/298/321/68/27 chars;
  one base-rate tokens=0 draw still captured 321 chars), and all 5 verified to sane, schema-identical
  verdicts — control `unsolved_goals`, t1 `sorry`, t1_shuffled `type_error`, bm25 + irrelevant
  `parse_error`, every row `success_audited=false`, `gen_host=ai04` preserved on every canonical row.
  The fan-out (6 balanced target shards under per-job `ollama serve` of goedel-prover-v2:32b, ≤6 cluster
  GPUs, ≤4/node) resumes off the 337 banked rows (each shard's `gen.jsonl` pre-seeded with its banked
  `(run_id,arm,target,draw)` rows so resume-skip never regenerates them); VERIFICATION + the C1 audit
  stay on-prem (`verify_deferred`, run_id-isolated APPEND into the canonical
  `confirm_v3_goedel32b/results.jsonl`, idempotent, `gen_host` preserved). The `--validate-banked`
  75/75 on-prem reproduction (verify half) and this prompt-equivalence + re-smoke (generation half)
  together establish the relocation is faithful. **STATUS: RESOLVED — fan-out launched.**

---
*Provenance: drafted 2026-06-11 ~00:45 during matrix50 v2 cell 3; slots filled on sweep
completion; signed and §10 execution record added 2026-06-11 ~07:25, before any confirmatory GPU
spend.*
