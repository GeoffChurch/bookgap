# Gap-deflation formulation

How to deflate one gap to get the next, satisfyingly/optimally. The
current pipeline has two inconsistent deflation models (greedy
hard-subtraction at seed time vs soft joint-norm + hand-tuned dedup
later); see `docs/superpowers/specs/2026-05-17-gap-deflation-bakeoff-design.md`.

Bake-off arms (all in `artifacts_scripts/gap_deflation_bakeoff.py`,
swappable via `deflate(cands, strategy, ...)`): `greedy_hard`
(baseline, byte-anchored to `kball_gap_probe._greedy_kball_gaps_inline`),
`greedy_geom` (residual-restricted pick), `submodular_fl` (sparse
k-ball facility-location), `dpp` (greedy MAP-DPP, candidate-pool
capped), `softmax_mc` (memory-bounded kNN-basin multicluster-softmax;
first build of `multicluster-softmax-membership`). Eval:
qualitative-primary leaderboards + inter-gap metrics + held-out-real-books
recall/spread (signal A) + synthetic-implant gate (signal C). Frozen
context: T10181, LIP, binary (T,F) graph (per
`docs/denoiser_bakeoff_results.md`).

## Status / working notes

- All 5 arms implemented + memory-bounded (no dense A×K anywhere),
  byte-stable harness reused; greedy_hard byte-anchored to the oracle.
- Observation worth following up at analysis time: DPP's Jaccard
  diversity degenerates when the k-ball radius is large relative to
  cluster separation (all candidate balls become near-identical, so
  the determinant has no diversity signal). This is expected to matter
  on the real L2-normalized sphere when books are sparse/far — flag
  when reading the real-run leaderboards.
- (living section — updated as the bake-off runs; migrates to
  `docs/gap_deflation_results.md` on completion)

### Cross-topic results (2026-05-18) — T10181 + T11303 + T10462

Held-out book recall@20 (greedy_hard / greedy_geom / submodular_fl / dpp):
- T10181 (NLP):        0.128 / 0.128 / **0.156** / 0.044
- T11303 (Bayesian):   **0.483** / **0.483** / 0.379 / 0.103
- T10462 (RL, sparse): 0.375 / 0.375 / **0.500** / 0.375

Conclusions (triangulated w/ qualitative leaderboards, all 0 errors):
- **greedy_geom ≡ greedy_hard** byte-identical on all 3 topics → the
  residual-restriction is inert at top-20. Drop greedy_geom.
- **submodular_fl is topic-dependent, NOT a universal winner**: wins
  T10181+T10462, *loses to plain greedy_hard on T11303*. The
  single-topic "submodular_fl wins" is overturned.
- **dpp robustly refuted for book-gaps**: worst/weakest recall + lowest
  spread on all 3; top gaps often non-book-depleted (high eff, p≈0.5,
  language clusters). Optimizes manifold coverage, not book-relevance.
- **softmax_mc** refuted on T10181 (over-fragments, artifact-prone);
  not re-run cross-topic.
- Original "greedy hard-subtraction → redundant gaps" premise NOT
  borne out: all arms' top-20 near-non-redundant (downstream
  refine+dedup handles it). Real differentiator = book-relevance of
  selection, where greedy_hard is already robust.

**Recommendation:** greedy_hard is the robust default (adequately
solves deflation; even wins T11303); submodular_fl = per-scope A/B
option, not a replacement; drop greedy_geom; don't use dpp/softmax_mc
for book-gaps. Caveats: proxy not LLM-coverage; T10462 spread
saturates (~ln of few held-out books); modest absolute recall —
triangulation not oracle.

**Status: investigation conclusive across 3 topics. Pending: sign-off
to migrate to `docs/gap_deflation_results.md` + transition this entry
per the validated/refuted convention.**

### 5-topic regime map (2026-05-18) — CONCLUSIVE

Added T10320 (Neural Nets, modern/article-dense) + T10820 (Fuzzy/Control,
classical/book-rich). greedy_hard vs submodular_fl held-out recall@20:

| topic | regime | g_hard | sub_fl | winner |
|---|---|---|---|---|
| T11303 Bayesian | classical/book-rich | 0.483 | 0.379 | g_hard +0.10 |
| T10820 Fuzzy/Ctrl | classical/book-rich | 0.457 | 0.326 | g_hard +0.13 |
| T10181 NLP | modern/article-dense | 0.128 | 0.156 | sub_fl +0.03 (marginal) |
| T10320 Neural Nets | modern/article-dense | 0.236 | 0.248 | ~tie +0.01 |
| T10462 RL | book-sparse | 0.375 | 0.500 | sub_fl +0.13 |

**Discriminating axis = subfield maturity / book density.** Classical
book-rich theory subfields → greedy_hard wins decisively. Modern
article-dense → wash. Book-sparse → submodular_fl wins decisively (its
only clear-win regime). Qualitatively all arms produce coherent
on-topic book-shaped gaps on all 5 (verified) → recall deltas are real
book-alignment, not artifacts. dpp worst/near-worst on all 5
(refuted for book-gaps). greedy_geom inert (byte-identical; dropped).

**FINAL recommendation:** greedy_hard = robust default (never
decisively beaten); submodular_fl = regime-specific, use ONLY for
book-sparse subfields; never dpp/softmax_mc/greedy_geom. Original
"greedy hard-subtraction unsatisfying" premise refuted at output
level. Caveat: proxy not LLM-coverage; a confirmatory LLM-coverage
run on a book-sparse topic (where submodular_fl wins) would be the
single highest-value pre-paper check.

**Status: superseded by the fl_bookdemand result below — do NOT
migrate on the 5-topic map alone; the recommendation has changed.**

### New arm: fl_bookdemand (2026-05-18) — beats both champions on 3/4 regimes

`fl_bookdemand` = facility-location over **book-anchored demand**:
customers = each training book's k=10 nearest articles (Nb);
`cov(g,b) = Jaccard(members_g, Nb[b])` (parameter-free, bounded —
chosen specifically so a giant background ball cannot saturate all
demand in one pick; an unnormalised count form *did* collapse and is
guarded by `test_fl_bookdemand_no_saturation_with_background`); greedy
FL maximises Σ_b max(0, cov(g,b) − best_cov[b]). Commit `9a7f037`.
Unifies greedy_hard's book-anchoring + submodular_fl's
coverage-spreading without a regime switch — hypothesis: wins
everywhere. Run vs the two regime champions, byte-anchored
(greedy_hard/submodular_fl rows reproduce the 5-topic table exactly).

| topic | regime | g_hard | sub_fl | **fl_bookdemand** | gaps_emitted |
|---|---|---|---|---|---|
| T11303 | classical/book-rich | 0.483 | 0.379 | **0.655** | 82 |
| T10820 | classical/book-rich | 0.457 | 0.326 | **0.609** | 128 |
| T10320 | modern/article-dense | 0.236 | 0.248 | **0.630** | 407 |
| T10462 | **book-sparse** | 0.375 | **0.500** | 0.125 | **30** |

**fl_bookdemand wins 3/4 decisively — beating BOTH prior champions on
their own turf** (T11303/T10820 were greedy_hard's decisive-win
regime; it also crushes the modern wash where neither champion broke
0.25). Far fewer, sharper, more book-aligned gaps (82–407 vs ~830;
highest unique_coverage@20 everywhere).

**But it does NOT win everywhere — it collapses on book-sparse
(T10462): 30 gaps, `heldout_spread = -0.0`, recall 0.125.** Structural,
not a bug: book-sparse → tiny book-anchored demand → FL exhausts it in
a few picks → held-out (also sparse) books not rediscovered. The
`gaps_emitted` column (added with this arm) makes the collapse
diagnostically unambiguous — 30 vs ~830 is the saturation signature.
Where books are too sparse to *define* a demand signal, the
non-book-anchored submodular_fl remains necessary.

**REVISED FINAL recommendation:** the universal winner is
**regime-gated, not single-method**. `fl_bookdemand` is the default
wherever book-density clears a threshold — it dominates classical
book-rich AND modern article-dense (the large majority of subfields),
superseding both greedy_hard and submodular_fl there.
`submodular_fl` remains the fallback ONLY for book-sparse subfields
where fl_bookdemand collapses. greedy_hard is no longer the
recommended default (beaten decisively on its own turf). Never
dpp/softmax_mc/greedy_geom. Open: where exactly is the book-density
gate (need a topic between T10320-dense and T10462-sparse to locate
it), and does a sparse-demand fallback inside fl_bookdemand (blend
toward submodular_fl as book-count→0) recover T10462 without losing
the 3 wins.

**Status: CONCLUSIVE on the regime split; recommendation changed
(fl_bookdemand-gated, not greedy_hard-default). NOT ready to migrate —
the book-density gate location + sparse-fallback question are the
natural next experiments before a results doc is frozen. Pending
sign-off.**

### fl_bookdemand_eb (2026-05-18) — emergent shrinkage; PROVISIONAL

Attempt to remove the regime branch: one FL objective over books ∪ a
data-driven k-means background coreset (`M = clip(A//k_book,64,4000)`);
shrinkage `M/(M+n_books)` emergent from the customer-count ratio, no
tuned λ. Spec
`docs/superpowers/specs/2026-05-18-fl-bookdemand-eb-shrinkage-design.md`;
arm commits `ba234d3`/`a04abcc`/`845d54a`; diag-instrument dual-module
fix `41ce35d`.

First run (held-out recall@20): T11303 0.483 · T10820 0.500 · T10320
0.285 · **T10462 0.500** (no gaps_emitted collapse: 730 vs
fl_bookdemand's 30). **EB fixes the book-sparse collapse but regresses
to ~submodular_fl everywhere — it sacrifices ALL THREE of
fl_bookdemand's standout wins** (0.655/0.609/0.630 → 0.483/0.500/
0.285). Apparent **over-shrinkage / book-rich dilution** (the exact
risk the spec's success criteria named): background customers dominate
even when books are plentiful because `M` floors at 64 and scales with
A regardless of book density. Caveat: the first run's diag readout was
blind (a `python -m` dual-module `_FL_DIAG_EB` split, fixed in
`41ce35d`); a confirmatory re-run with valid per-topic shrinkage
logging is in progress (`flbdeb2_*`) before this is certified.

**CONFIRMED (diag-valid re-run `flbdeb2_*`, scorecard
deterministically reproduced).** Per-topic diag:

| topic | n_books | M | bg:book | shrinkage | EB | fl_bookdemand |
|---|--:|--:|--:|--:|--:|--:|
| T11303 book-rich | 116 | 1412 | 12:1 | 0.924 | 0.483 | 0.655 |
| T10820 book-rich | 183 | 2173 | 12:1 | 0.922 | 0.500 | 0.609 |
| T10320 modern | 660 | 4000* | 6:1 | 0.858 | 0.285 | 0.630 |
| T10462 sparse | 34 | 1278 | 38:1 | 0.974 | 0.500 | 0.125 |

(*M hit the 4000 cap.) Shrinkage moves the *right direction* with
book density (0.858 modern-dense → 0.974 sparse) but the operating
point is uniformly background-dominated: `M = clip(A//k_book,64,4000)`
keys the prior resolution to **article** density, which is high for
every topic, so `n_books ≪ M` always (6:1–38:1) and shrinkage stays
0.86–0.97 even book-rich. Result: EB ≈ `submodular_fl` everywhere —
collapse fixed, all 3 fl_bookdemand wins forfeited.

**Diagnosis:** the regularization strength must be a function of the
book signal's *reliability* (`n_books`), but it is currently a
function of *corpus size* (`A`). Emergent-shrinkage-from-customer-mix
is sound in principle (it moves the right way) but cannot interpolate
to books-dominant while `M ⟂ n_books`. Concrete levers, in order of
directness: (a) key the prior resolution to book density —
e.g. `M ≈ n_books · (1/s* − 1)` for a target shrinkage `s*` (makes the
bg:book ratio, hence shrinkage, a tuned constant independent of
regime); (b) `eb_book_weight ∝ A/n_books` (up-weight scarce books
instead of shrinking M); (c) target a fixed shrinkage `s*` and solve
for M per topic. (a)/(c) are the same idea (pin the operating point);
they convert the *emergent but mis-anchored* ratio into a *controlled*
one while keeping the no-regime-branch property.

**Status: CONCLUSIVE — emergent-shrinkage-as-specified is refuted as a
universal winner (over-regularizes: prior keyed to A not n_books).
Regime-gated fl_bookdemand remains the standing recommendation. The
operating-point fix (lever a/c) is the natural next experiment and is
a one-parameter change, not a redesign. Pending sign-off on whether to
pursue it or freeze the regime-gated result.**
