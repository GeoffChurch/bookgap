# FF stats phase 3: pilot-based joint calibration of K_query and member subsample

## Status (updated 2026-05-14)

**Partially validated.** The calibration mechanism is implemented in
`artifacts_scripts/gap_kball_llr_refine.py:_calibrate_stats_params`
and runs cleanly. Empirical behaviour on a 76-topic subfield-scope
deployment:

- **Member-subsample side: works.** 56/76 topics picked m < all
  (most commonly m=100-200, sometimes m=25-50). For saturator gaps
  with k\*≈1280 members, this is a real 6-12× reduction in HNSW
  queries per gap.
- **K-query side: never picks below ceiling at default tolerance.**
  75/76 topics picked K=1500 (the ceiling of the candidate grid);
  only 1 picked K=800. Pilot p95|Δ| at K=800 m=all hovers around
  0.4-0.9 logits, far above the δ=0.1 default threshold. The
  K=500 baseline cited in the original "expected savings" section
  doesn't reproduce under the calibration's own statistical
  criterion at δ=0.1 — and small-K candidates can have catastrophic
  max|Δ| (60-70 logits) on saturator gaps where truncating the NN
  list materially changes median_logit.
- **Net realised speedup: ~3-5× (from m-side only)**, not the ~10×
  the original joint-savings framing predicted.

## What this means for the LLR_REFINED integration

Shipping `--k-query-stats=auto` as the default is **safe and worth
keeping** because the m-side delivers real savings without quality
loss. But the design doc should not claim K-side adaptivity as a
benefit, and the comparison to "the K=500 baseline" should be
treated as research provenance, not as a guarantee the calibration
will achieve it.

## Paths forward (pick one if K-side savings are worth chasing)

1. **Loosen δ to ~0.3 logits.** Empirically K=800 m=all passes that
   threshold on most topics. Risk: small-K candidates' catastrophic
   failure on saturator gaps would also start passing. Mitigation:
   keep δ tight on `max|Δ|` while loosening on `p95|Δ|`.
2. **Replace global δ with a per-gap k\*-adaptive K.** Saturator
   gaps need K=1500; small gaps could tolerate K=200. Stratify the
   pilot by k\* decile and pick K per decile.
3. **Accept m-only adaptivity.** Document the K-side as fixed at
   1500 by default; remove the K-grid from the calibration grid;
   drop the `--calibrate-k-grid` flag. Saves the K-side pilot cost
   (~30-60s per topic) for no quality loss.

Recommendation: **option 3** unless someone has a concrete reason to
chase K-side savings. The calibration's pilot cost is non-trivial
and the K-side pilot is doing no useful work at the current default
tolerance.

## Cross-references

- Implementation: `_calibrate_stats_params` at
  `artifacts_scripts/gap_kball_llr_refine.py:932`.
- Subfield deployment evidence:
  `cache/.jobs/per_topic_17_1702_1778677820.log` — grep for
  `\[calibrate\] CHOSEN` to see the per-topic verdicts.
- Member-subsample side validation:
  `cache/llr_refine_kquery_test/T{11303,10181}_v2auto.csv` and
  the manual sweeps `T11303_kqs{200,500}.csv`.
- Stein two-stage analogue for null subsampling:
  `_compute_corpus_null_stein` in the same file (this *does* work
  as advertised).

## Original idea (preserved for context)

[The original brief proposed joint (K, m) calibration with claimed
~10× speedup. The mechanism was implemented; the m-side claim held,
the K-side claim did not at the default tolerance. See "Status"
above for the current understanding.]
