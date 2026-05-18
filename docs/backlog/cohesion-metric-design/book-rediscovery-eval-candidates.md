# Book-rediscovery eval — candidate experiments

**Living tracking doc.** The held-out-real-books rediscovery eval
(signal A: mask 20% of books before detection, measure
`heldout_book_recall@K` + `heldout_spread`; implemented in
`artifacts_scripts/gap_deflation_bakeoff.py` —
`heldout_book_split` / `heldout_recall_spread`) is a *new external
evaluation axis*: prior gap/denoiser work was judged on LLM-coverage %,
geometric/significance proxies, qualitative reads, and templated-leak
artifact-recall — **none measured "does this method rediscover where
humans actually wrote books, non-redundantly."** This doc enumerates
everything worth running through that eval so we don't lose track. Keep
it updated as items move to in-progress / done / refuted.

## Why this axis is different (don't lose this framing)

Recall/spread of held-out real books is cheap, objective-ish, and
orthogonal to every prior proxy. Conclusions resting on the weaker
proxies (esp. `docs/denoiser_bakeoff_results.md`) should be re-checked
here. It is *triangulation, not an oracle* — modest absolute recall is
expected; treat as one signal among qualitative-primary + inter-gap.

## Candidates (priority order)

### P1 — Denoiser bake-off + held-out (highest leverage)
Add `heldout_book_recall@20` / `heldout_spread` columns to the denoiser
bake-off scorecard and re-run. The LIP-binary win, LIP/LPP
mirror-image, and "PCA significance is artifact-driven" conclusions in
`docs/denoiser_bakeoff_results.md` rest on sig_count/four-corner/jaccard
— the *weaker* proxies. Tests whether they hold on the book-relevance
axis. Cheapest high-value item: the held-out harness already exists and
is reusable; small scorecard change + a re-run (T10181 ± T11303/T10462,
the byte-stable hnsw config). **Status: proposed (awaiting go-ahead).**

### P1 — Soft-EM (article-as-seed) through the held-out harness
The cohesion-aware production pipeline (`docs/gap_method_comparison.md`):
moving-centroid HNSW, soft-membership posteriors, logit_sum ranking.
Substantively different gap-detection method vs K-ball; carves far
finer (≈423 basins vs K-ball's 200); only ever compared qualitatively
+ on artifact-recall. Held-out recall/spread directly adjudicates the
open question: does fine carving *rediscover more distinct human-book
regions*, or just fragment? Emits the same gap-dict shape → slots into
the harness. **Status: proposed.**

### P2 — Deflation arms, cross-topic (in progress)
`submodular_fl` won the T10181 triangulation (best held-out recall
0.156 / spread 3.39). Cross-topic baselines **complete** (5 topics):
greedy_hard wins book-rich/classical (T11303 0.483, T10820 0.457),
submodular_fl wins book-sparse (T10462 0.50) and is marginal on modern
(T10320 0.248); dpp consistently worst book-alignment everywhere —
confirms the regime axis in `gap-deflation-formulation.md`. **New arm
`fl_bookdemand`** (facility-location over book-anchored demand; Jaccard
coverage, anti-saturation; commit `9a7f037`) launched 2026-05-18 vs
greedy_hard + submodular_fl on the regime-spanning set
(T11303/T10462/T10320 running, T10820 queued; jobs `flbd_*`) to test
the "wins everywhere" hypothesis. **Result (2026-05-18):**
fl_bookdemand wins 3/4 decisively — T11303 0.655, T10820 0.609,
T10320 0.630 — *beating both prior champions on their own turf* — but
**collapses on book-sparse T10462 (recall 0.125, 30 gaps emitted,
spread −0.0)**. Not a universal winner; the recommendation is now
**regime-gated**: fl_bookdemand default above a book-density gate,
submodular_fl fallback below it. Next: locate the gate (a topic
between T10320-dense and T10462-sparse) + try a sparse-demand fallback
inside fl_bookdemand. **`fl_bookdemand_eb`** (emergent-shrinkage,
no-branch) tested cross-topic: fixes the book-sparse collapse but
over-regularizes (prior resolution `M` keyed to article density not
book density → bg:book 6:1–38:1, shrinkage 0.86–0.97 everywhere → ≈
submodular_fl, all 3 fl_bookdemand wins forfeited). Refuted as a
universal winner; the operating-point fix (key M / book-weight to
`n_books`, not `A`) is the open one-parameter follow-up. **Status:
EB conclusive (refuted as-specified); regime-gated fl_bookdemand
stands. Feeds `gap-deflation-formulation.md`.**

### P2 — Denoiser × deflation interaction
Both bake-offs froze the *other* axis (deflation frozen at
LIP+binary; denoiser frozen at greedy_hard deflation). Held-out makes
a small targeted cross cheap: e.g. {LIP, none, lpp} × {greedy_hard,
submodular_fl} on T10181. Tests whether the best deflation depends on
the denoiser (and vice versa). **Status: proposed.**

### P2 — Dual-landmark held-out (hold out chapters, not books)
`effective_size_books`/`_chapters` already exist. Symmetrically hold
out 20% of `book-chapter`-type works and measure chapter-rediscovery.
Directly tests the "covered by chapters, not books" axis on the same
external methodology — the paper's open book-vs-chapter question.
**Status: proposed.** (See `dual-landmark-eff-size-reporting.md`.)

### P3 — p-ball radius_fraction sweep under held-out
The paper uses `p=3/4` K-balls and flags "choice of p" as future work.
Held-out rediscovery gives an objective way to pick `p`: sweep
`radius_fraction ∈ {0.5, 0.75, 1.0, ...}` and read recall/spread.
Cheap re-screen of an existing free parameter. **Status: proposed.**

### P3 — Held-out eval methodology hardening (meta)
Sensitivity of signal A to its pinned thresholds (20% mask seed,
≥50% of 10-NN coverage, K=20). Spec pinned these deliberately
(not tuned to outcomes); a sensitivity sweep tells us how robust the
ranking is to them before any conclusion is enshrined. **Status:
proposed.**

## Explicitly NOT worth re-trying under this eval

Recorded so the "should we re-try X?" question has a standing answer
(`docs/dead_ends/`): **batch-independent ADD** (runaway 2400-article
clusters), **adaptive-K** (refuted, commit `dc30967`),
**centroid-Bayesian likelihood** (degenerate in 384-d via the
`−(d/2)logσ²` term), **inverse-square kernel** (1/r^(d−2) ≈ delta at
d=384), **monotone-score-as-boundary** (`effective_size`/`logit_sum`
monotone in size). These were refuted for *intrinsic*
mathematical/algorithmic reasons the held-out metric cannot overturn —
re-running them is re-running a known-broken experiment.

## Cross-cutting caveats (apply to every item above)

- Single-topic conclusions don't generalize — always ≥2 topics.
- Proxy, not LLM target-gap-coverage (spec §8 non-goal); a winner
  still warrants one confirmatory LLM-coverage run before the paper.
- Held-out books are sparse + AI-topic-biased; recall is a coverage
  proxy. Triangulate with qualitative-primary + inter-gap metrics.
- Reproducibility: use the byte-stable `_build_topic_index`
  efc=512/efs=2000 (or `--index-type flat` for exact).
