# Funding "waste" via the inverse gap score (high $ / low output) — REFUTED: dollars and output are positively coupled; what's left is just grant age

**Tested 2026-06-15** (gap-detection domain-application spike; scripts
`artifacts_scripts/funding_spike/{collect.py,analyze.py}`, NIH RePORTER + iCite,
n = 3000 R01 grants, first FY 2013–2016, dollar-neutral subsample).

Context: brainstorming where the gap method could transfer beyond articles/code. The gap score is
Δ = log(structural) − log(operational). "Funding waste" was pitched as the **sign-flipped gap** — the
"two signs of one Δ" framing (parallel to OSM completeness-vs-anomaly): operational = dollars awarded
(HIGH), structural = research footprint produced (LOW: few/no linked publications, low citations/RCR)
→ flag well-funded grants that produced little, as **audit triage** (never accusation).

**Expected:** a recoverable population of high-dollar / low-output grants — the inverse extreme of the
gap score, analogous to how high-structural / low-operational surfaces genuinely under-tested code.

**Actual:** there is essentially no such population once you control for the obvious confound.
- Raw dollars→output is **positive, not negative**: Pearson r(log$, log output) ≈ **+0.18** (pubs),
  +0.14 (citations), +0.17 (RCR), all p ≪ 1e-10. Only **1.2%** of grants have zero publications.
- Controlling for grant **age** (+ cohort FY + duration), the partial dollars→output correlation
  shrinks toward zero but stays **non-negative** (+0.07 pubs; +0.03 citations, n.s.; +0.07 RCR).
  **Age dominates the OLS** (t ≈ 33 for age vs ≈ 4 for log-dollars; age alone → R² ≈ 0.30).
- Restricting to mature grants (age ≥ 15): zero-pub **collapses to 0.7%**, and dollars→output is
  positive again. Zero-pub grants are systematically *younger* (mean age 13.0 vs 17.2), not bigger.
- The top "funded-but-invisible" flags are **legitimate** late/non-publishers even within R01 —
  clinical/device trials (results report to ClinicalTrials.gov, not always PubMed-linked),
  vaccine/product development, behavioral interventions — plus imperfect grant↔PubMed linkage. Not duds.

**Diagnosis:** the inverse-Δ fails for a structural reason that distinguishes this domain from code:
in NIH funding **dollars and output are positively coupled** (more money → modestly *more* footprint),
so there is no high-$/low-output corner to mine — unlike code, where structural centrality genuinely
*dissociates* from test coverage. What remains of the "waste" extreme is almost entirely the
**time-lag confound**: the raw inverse score just re-measures grant *age* (recent grants haven't
published yet). Same failure mode as monotone-in-size scores
([monotone-scores-as-boundary-criteria](../cohesion-metric-design/monotone-scores-as-boundary-criteria.md))
and the code size-confound (Δ partly proxies class size, `docs/code-gap-triangulation.md` §size-control)
— a raw extremal score proxying a nuisance variable — **except here, controlling the nuisance leaves
nothing, not a residual signal.** The "two signs of one Δ" framing only works when the two axes
*dissociate*; here they don't.

**Salvage:** the only defensible residual is an age×mechanism-stratified analysis restricted to mature
grants, triaging the extreme ~0.7% zero-output tail with per-grant publication-linkage verified by hand
— but the age-controlled partial correlation is non-negative, so expected yield is ≈ noise. Funding
*waste* (inverse gap) is exhausted as a standalone signal. **Not refuted here:** the funding *gap*
(structural-central-but-under-funded topics, the *forward* Δ) is a separate idea — but it shares a
**soft-oracle** problem (no ground truth for "should have been funded," and the universal
temporal-negative warns that under-funded central topics may be correctly out-of-fashion, not future
winners). The clean external oracle that makes OSM completeness attractive has no analog here.
Separately, real fraud/waste detection (collusion, conflict-of-interest) lives on a *different graph*
(bidders↔evaluators↔beneficiaries) owned by deployed tools (EU **ARACHNE**, World Bank) — unreachable
by the concept/citation method, and high-stakes enough that "audit triage, never accusation" is the
only responsible framing regardless.
