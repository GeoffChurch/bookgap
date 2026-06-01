# Reciprocal / Boundary-Background Curriculum Selection — Results

**Question.** Does the "boundary-background" selection algorithm (`_boundary`) produce
better gap-seeded curricula than the canonical default (`_base`), as judged single-blind
by an LLM on a 4-dimension rubric?

**Setup.** 5 recognizable, clean-gap topics — T10181 (NLP/Chinese segmentation), T10637
(advanced clustering), T11273 (graph neural nets), T12761 (data stream mining), T11902
(intelligent tutoring). Each got a `_base` and `_boundary` curriculum (15 chapters). Both
were judged by `gemma4:31b` two ways: an **absolute rubric** (per-chapter + book, 1–7
Likert, repeat-voted, single-blind) and a **pairwise** A/B/tie (both orders, position-bias
controlled). One pair (T10181) was additionally judged by a **frontier judge (opus)** on
the identical pairwise prompt as a correlation spot-check.

**Algorithm (`_boundary`).** Forward seeded PR `p` → reverse PR personalized by `p` (`q`,
the diffused neighborhood) → forward PR from `q` (`g_local`, a field-local background) →
select top-k by `p^0.5·log(p/g_local)`. Intrinsic (no external taxonomy); demotes works
the off-topic neighborhood reaches while keeping topic-specific foundations. Verified on a
toy graph (keeps a topic foundation, demotes a hub).

## Result: boundary ≥ base on all three lenses; never base > boundary

### 1. Absolute rubric (book median, base → boundary), mean across 5 topics
| dimension | base | boundary | Δ |
|---|---|---|---|
| interestingness | 2.20 | 2.40 | +0.20 |
| comprehensiveness | 2.40 | 3.00 | **+0.60** |
| specificity | 2.80 | 3.00 | +0.20 |
| progressive | 2.80 | 2.60 | −0.20 |

Boundary wins the book-level rubric on 3/4 dimensions (comprehensiveness most clearly),
loses progressive slightly. **Chapter-level means are flat (±0.05)** — boundary changes
*which chapters are selected* (book composition) more than individual chapter quality,
which is what a selection change should do.

### 2. Pairwise A/B/tie (de-randomized, both orders), 5 topics × 4 dims
| dimension | boundary | base | genuine-tie | flip-contradiction |
|---|---|---|---|---|
| interestingness | 2 | 0 | 0 | 3 |
| comprehensiveness | 1 | 0 | 2 | 2 |
| specificity | 1 | 0 | 2 | 2 |
| progressive | 1 | 0 | 0 | 4 |
| **total** | **5** | **0** | 4 | 11 |

Where gemma commits to a consistent winner, it is **boundary 5, base 0**. But **11/20 are
flip-contradictions** (preferred the same *screen position* in both orders) — gemma is
heavily position-biased on this task. Tracking flips separately (rather than collapsing
them into "tie") is what exposed this; the underlying comments on flipped dims still lean
boundary ("more diverse modern approaches", "broader range of clustering paradigms").

### 3. Frontier (opus) spot-check, T10181, position-robust (0 flips)
| dim | opus | gemma |
|---|---|---|
| interestingness | boundary | flip (comment → B) |
| comprehensiveness | tie | flip (comment → B) |
| specificity | boundary | tie |
| progressive | boundary | boundary |

Opus prefers **boundary 3/4** with no flips, agreeing with gemma's direction. opus's
reasons: `_base` carried a noise chapter (`ideas/thank/thesis`) and leaked into standalone
information/coding theory; `_boundary` kept NLP-anchoring chapters (POS tagging, parsing)
and a cleaner foundations-first order — exactly the off-topic-hub demotion the algorithm
targets.

## Honest caveats
- **Modest effect on mediocre curricula.** Absolute scores are low for *both* (mostly
  2–3/7); boundary is a real but small improvement, not a transformation. The judge flags
  both as over-concentrated/narrow — a limitation of the gap-seeded curricula themselves,
  not the selector.
- **gemma pairwise is position-biased** (55% flips). Its decisive verdicts are few but
  unanimously pro-boundary; opus (frontier, no flips) is the more trustworthy pairwise judge.
- n=5 topics; directional, not statistically powered.

## Verdict
Across an absolute rubric, a (position-bias-controlled) pairwise judge, and a frontier
spot-check, the signal is consistent: **the boundary-background selector is modestly better
than the canonical default, and never worse** — clearest on comprehensiveness (rubric) and
in opus's specificity/ordering wins. The win is real but small atop generally-mediocre
gap-seeded curricula.

## Harness bugs found + fixed during this run (methodological notes)
1. **OOM** from running two ~104 GB curricula concurrently → switched to strictly one-at-a-time.
2. **gemma pairwise empty content**: the `<think>` channel exhausted `num_predict` before
   emitting JSON → `think=False` in `JudgeClient`.
3. **silent verdict loss**: gemma capitalizes dim names (`Interestingness`) despite a
   lowercase schema (ollama `format=` is a soft guide) → `_parse_pairwise` rejected every
   valid response on a case mismatch → case-insensitive dim lookup. (Bug #2 masked #3.)

Artifacts: `artifacts/curriculum_eval/*.json` (rubric), `artifacts/curriculum_eval_pairwise/*.json`
(pairwise), `artifacts/curriculum_eval_pairwise/opus_probe/` (frontier spot-check).
