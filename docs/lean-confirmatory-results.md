# Lean curriculum-context — CONFIRMATORY RESULTS (pre-reg v2 §5)

*Run `confirm_v3_goedel32b`, goedel-prover-v2:32B Q4_K_M, Mathlib.Probability.Kernel
frame v3 (n=125 GT-gated targets, 3×3 strata). 1,875 rows = 5 arms × 125 targets × 3 draws,
**all generated and all verified** on the canonical Lean+Mathlib v4.20.0 toolchain. Analysis
per the SIGNED pre-registration `preregistration-curriculum-context-v2.md` §5–§6. Reproduce:
`PYTHONPATH=. conda run -p .conda python -m curriculum_lean.confirm_analysis`.*

## TL;DR

**Relevant premise statements in context lift a 32B prover's audited success ~6× over no
context — but the lift is "premise retrieval," not "curriculum."** Dependency-graph selection
(t1) is statistically indistinguishable from plain BM25 lexical retrieval, and proof order
doesn't matter. The effect is real, content-driven (not warmth), and mechanistically mediated
by the prover citing the surfaced premises — it just isn't *curriculum-specific*.

- **H1 (curriculum > retrieval): NOT supported in its discriminating claim.** t1 ≫ control
  (decisive) but t1 ≈ bm25 (tied).
- **H2 (ordering): FALSIFIED.** t1 ≈ t1_shuffled — selection, not sequence, carries the effect.
- **H3 (content vs warmth): SUPPORTED, airtight.** t1 ≫ irrelevant; placebo successes never
  cite the context, real-arm successes almost always do.
- **H4 (capability-relativity): NO DATA** — the gemma secondary (`confirm_v3_gemma_off`) was
  never generated (cluster relocation ran 32B-only). One decision pending (below).
- **H5 (difficulty model): SUPPORTED.** Audited success decreases with total premise surprisal
  (Spearman −0.28, p=0.001), in the pre-committed direction.

## Primary — audited pass@3 per arm (the registered metric)

Audited frame: success = verifier-accepts **and** the constant-dependency audit confirms the
proof never references the withheld target. Circular/unaudited draws demoted to failure.

| arm | pass@3 | rate | Wilson 95% CI |
|---|---|---|---|
| control (no context) | 3/125 | **2.4%** | [0.8%, 6.8%] |
| **t1** (curriculum stmts, top_n=4) | 19/125 | **15.2%** | [10.0%, 22.5%] |
| t1_shuffled (same set, shuffled) | 18/125 | 14.4% | [9.3%, 21.6%] |
| bm25 (equal-K lexical retrieval) | 17/125 | 13.6% | [8.7%, 20.7%] |
| irrelevant (equal-K placebo) | 4/125 | 3.2% | [1.3%, 7.9%] |

The three **relevant-context** arms cluster at ~14–15% (≈6× control); **control and the placebo**
cluster at ~2–3%. The dividing line is *relevant content present vs absent* — nothing finer.

## Exact McNemar (paired, within-target)

| contrast | discordant (+win/−loss) | McNemar p | Δ | bootstrap 95% CI |
|---|---|---|---|---|
| t1 vs control | +17 / −1 | **<0.001** | +12.8% | [+6.4%, +19.2%] |
| t1_shuffled vs control | +15 / −0 | **<0.001** | +12.0% | [+6.4%, +18.4%] |
| bm25 vs control | +16 / −2 | **0.001** | +11.2% | [+4.8%, +17.6%] |
| irrelevant vs control | +2 / −1 | 1.000 | +0.8% | [−1.6%, +4.0%] |
| **t1 vs bm25** | **+14 / −12** | **0.845** | +1.6% | [−6.4%, +9.6%] |
| **t1 vs t1_shuffled** | **+5 / −4** | **1.000** | +0.8% | [−4.0%, +5.6%] |
| t1 vs irrelevant | +16 / −1 | **<0.001** | +12.0% | [+5.6%, +18.4%] |

## Hypothesis verdicts (§6)

**H1 — curriculum context beats no-context AND lexical retrieval. NOT SUPPORTED (discriminating
claim).** t1 ≫ control is decisive (+12.8pp, p<0.001). But t1 vs bm25 is a near-perfect wash:
+1.6pp, discordant +14/−12, p=0.85, CI spans zero. By the *letter* of the registered falsifier
("falsified if t1 − bm25 ≤ 0") H1 is not technically falsified — t1 is nominally +1.6pp above
bm25 — but that margin is indistinguishable from zero. The substantive reading is exactly the
consequence the pre-reg attached to this outcome: **"curriculum" here is just premise retrieval.**
Graph-centrality selection buys nothing over equal-budget BM25.

**H2 — ordering. FALSIFIED.** t1 vs t1_shuffled: +5/−4, p=1.00. The selected *set* carries the
whole effect; its *sequence* is inert. (Consistent across pass@1/2/3.)

**H3 — content vs warmth. SUPPORTED (airtight).** t1 vs irrelevant: +16/−1, p<0.001; and the
placebo is itself indistinguishable from control (+2/−1, p=1.00). The lift requires *relevant*
content, not merely the presence of *some* context. The waypoint mechanism (below) seals it.

**H4 — capability-relativity (secondary). NO DATA.** The registered gemma secondary
`confirm_v3_gemma_off` (2 statement arms × 125 × 3 ≈ 750 rows) was never run — when the sweep was
relocated to the Stony Brook cluster it carried only the 32B primary. To evaluate H4 we'd need
~4h of gemma GPU (control + t1). **Decision pending** — see below.

**H5 — difficulty model (observational). SUPPORTED.** Per-target audited success-rate vs total
premise surprisal `s_sum_term`: Spearman **−0.283, p=0.001** (Kendall −0.224, p=0.001), in the
pre-committed negative direction (harder = rarer premises = less success). It is specifically the
*sum* of surprisal, not the mean (`s_mean_term` ρ=+0.007) — total toolkit rarity is the difficulty
axis, consistent with `n_tactics` (−0.24) and `proof_chars` (−0.21). *Caveat:* the committed
`validate_difficulty` tool reports this as null because its primary aggregate is the **mean graded
ordinal** (pooling sorry/unsolved partial credit), which dilutes the audited-success signal; H5's
registered outcome is audited *success* specifically, on which the relationship holds.

## Secondary metrics

| arm | circular-close rate | tokens / solved target | waypoint: successes citing a surfaced name |
|---|---|---|---|
| control | 8/375 | 776,394 | n/a (no surfaced names) |
| t1 | 3/375 | **129,440** | **29/30 (97%)** |
| t1_shuffled | 6/375 | 132,962 | 35/37 (95%) |
| bm25 | 8/375 | 163,157 | 37/39 (95%) |
| irrelevant | 4/375 | 642,268 | **0/9 (0%)** |

- **Waypoint usage is the mechanistic smoking gun.** When a relevant-context arm succeeds, the
  proof cites a surfaced premise 95–97% of the time. When the *placebo* succeeds (9 draws), it
  cites a surfaced name **0%** of the time — those are the prover proving despite the context,
  numerically matching control's intrinsic rate. Content arms work *because* the prover uses the
  premises; the placebo's rare wins are unrelated to its context. This is the causal core of H3.
- **Instrument health: clean.** 29 circular draws total across 1,875, all correctly demoted (the
  audit is doing its job — note control's 8 circulars are pure name-recall with no context). At
  the target level, **zero** arm has a circular as its only "success": every pass@3 number above
  is audited-clean. t1 has the *fewest* circulars (3) — curriculum context does not induce
  name-recall cheating.
- **Token efficiency** tracks success: t1/bm25 spend ~130–163K tokens per solved target; control
  and the placebo spend 5–6× more (they solve so little). t1 is nominally the most efficient, but
  on tiny solved counts this is noisy.
- *Exploratory* graded partial-progress (best ordinal/target): t1 leads on mean ordinal (1.26 vs
  control 0.98) and on share reaching "typechecks-with-sorry" (45% vs 31%) — the same ordering as
  pass@3, gently.

## What this means

The pre-registered question was whether *curriculum-derived* context (dependency-graph centrality
selection) beats the alternatives. The answer is **no, not over lexical retrieval** — but the
broader and more robust finding is that **a handful of relevant premise *statements* is what helps
a strong prover**, and how you obtain them (graph centrality vs BM25) and how you order them does
not matter at this budget. This converges with — and sharpens — the curriculum-elaboration
negative result (`lean-curriculum-bidir-and-topics`): bidirectional PR, depth, and diversification
all *hurt or null*; here even the base graph-selection itself doesn't separate from retrieval. The
prover wants *a few tight, directly-relevant theorem premises*; the selection machinery beyond
"relevant" is not paying for itself in this Mathlib-probability frame.

H3 + the waypoint mechanism make the *content* claim airtight, and H5 gives a working
within-family difficulty law (total premise surprisal). H1's narrowing ("selection, not
curriculum") and H2's falsification ("set, not order") are themselves clean, publishable results —
the claim just narrows from "curriculum helps" to "relevant premises help, however retrieved."

## Open decision — H4

H4 (does the t1−control delta grow with prover capability?) is the one registered hypothesis with
no data. Running it = ~4h of local gemma GPU for `confirm_v3_gemma_off` (control + t1, 125×3 each).
The GPU is currently occupied by the data-science r4 book build; H4 would queue behind it. Options:
fire it after the book frees the GPU, or report H4 as not-evaluated and close the experiment on the
primary. (The markov-categories book — the other half of the original task — is intentionally
**not** launched.)
