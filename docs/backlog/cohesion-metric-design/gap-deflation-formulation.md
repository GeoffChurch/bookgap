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

**Operating-point fix tested (n_books-keyed s∈{.3,.5,.7} variants;
arms commit `0d2da2e`).** `M = n_books·s/(1−s)` pins shrinkage≈s
A-invariant (verified). 7-arm × 4-topic frontier sweep
(`ebsw_*`, scorecards reproduced byte-identical → trustworthy).

Recall@20 vs shrinkage (s≈0 = fl_bookdemand, s≈1 = submodular_fl,
s≈0.9 = A-keyed fl_bookdemand_eb):

| topic | n_books | s≈0 | s30 | s50 | s70 | s≈.9 | s≈1 | g_hard |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| T11303 book-rich | 116 | **.655** | .621 | .310 | .448 | .483 | .379 | .483 |
| T10820 book-rich | 183 | .609 | **.630** | .565 | .457 | .500 | .326 | .457 |
| T10320 modern | 660 | **.630** | .588 | .455 | .400 | .285 | .248 | .236 |
| T10462 sparse | 34 | .125 | .375 | **.500** | .375 | .500 | .500 | .375 |

**No constant s is a clean universal winner — regime optima diverge.**
Book-rich/modern peak at low shrinkage (s≈0–.3, monotone decline as s
rises); book-sparse T10462 is an inverted-U peaking at s≈.5 (low s
collapses to .125). s50 rescues T10462 but craters T11303 (.655→.310);
s30 is near-ceiling on the 3 dense topics but leaves T10462 at .375.

**Two strong results:** (1) **s30 is an excellent single no-branch
method** — ~95% of the dense ceiling (.621/.630/.588; *beats* pure
fl_bookdemand on T10820) AND converts the T10462 collapse .125→.375,
zero regime `if`. Below the *gated* champion only on book-sparse
(.375 vs submodular_fl .500). (2) **The universal-winner schedule is
now empirically pinned**: optimal shrinkage is a monotone function of
book density — ≈.2 dense, ≈.5 sparse, both endpoints measured. A
one-function `s = f(n_books/A)` schedule would plausibly dominate
fl_bookdemand on dense AND submodular_fl on sparse: a no-branch
universal winner designed from data, not guessed.

**Status: CONCLUSIVE. Emergent-constant-shrinkage refuted as a
universal winner, but the frontier is mapped and yields (a) s30 as a
strong no-branch default and (b) a data-grounded shrinkage-schedule
design. Standing recommendation: regime-gated fl_bookdemand OR s30
(no-branch, near-equivalent). Next experiment (one-function, not a
redesign): the book-density shrinkage schedule. Pending sign-off:
pursue the schedule, adopt s30 and freeze, or keep regime-gated.**

### fl_bookdemand_resid (2026-05-18) — intrinsic, no knob, no fit — TERMINAL

A hardcoded `s=f(book-density)` schedule would overfit (0.5 rests on
one sparse topic; calibrating to held-out recall is circular). The
intrinsic answer: background coreset over **book-orphan articles
only**, `M_bg = clip(|O| // max(k_book·n_books, k_book), 0, 4000)` —
operating point set by orphan density *per book*, zero tuned
constants, no held-out knowledge. Arm commits `744b93b`/`6d845fc`
(code review caught + fixed an A-keyed-equivalence defect:
`|O|//k_book` reduces to the refuted A-keyed EB for clustered books;
per-book normalization fixes it). 5-arm × 4-topic
(`resid_*`, baselines reproduced byte-identical).

| topic | n_books | orphan_frac | M_bg | fl_bd | s30 | **resid** | sub_fl |
|---|--:|--:|--:|--:|--:|--:|--:|
| T11303 book-rich | 116 | .934 | 11 | **.655** | .621 | .621 | .379 |
| T10820 book-rich | 183 | .940 | 11 | **.609** | .630 | .565 | .326 |
| T10320 modern | 660 | .934 | 10 | **.630** | .588 | .618 | .248 |
| T10462 sparse | 34 | .975 | 36 | .125 | .375 | .375 | **.500** |

Diag confirms the mechanism is exactly right: orphan_frac ≈ .93–.98
everywhere (books cluster, never blanket — as predicted) yet per-book
normalization keeps M_bg tiny → shrinkage ≈ .06–.09 dense, ≈.51
sparse (the projected universal-winner profile). It cleanly escaped
the A-keyed failure.

**But it is NOT a universal winner.** Near-matches fl_bookdemand on
dense (.621/.565/.618 vs .655/.609/.630; on T10320 numerically
*identical*) and fixes the sparse collapse (.125→.375, healthy
spread, no gaps collapse) — yet falls ~.03–.04 short of the dense
ceiling AND leaves book-sparse T10462 at .375, **0.125 below
submodular_fl's .500**. With only 34 books, pure-manifold coverage
beats *any* book-anchored-demand construction however the background
is regulated.

**FINAL CONCLUSION (whole investigation): the book-vs-manifold
tension is FUNDAMENTAL, not a bad-knob artifact.** No single
no-branch demand objective — A-keyed, constant-s, or
intrinsic-residual — recovers both the full dense wins AND
submodular_fl's sparse .500. fl_bookdemand_resid is the best
*principled* no-branch method (intrinsic, zero fitted constants,
mechanism validated) and is the scientifically preferable no-branch
default over the fitted s30 (≈equal numbers, no magic constant).
Regime-gated fl_bookdemand/submodular_fl still wins the absolute
numbers (full dense + .500 sparse) at the cost of the branch.

### Best-of-both reframes (2026-05-18) — A union, B gen→rank — TERMINAL+

Escape the "one mixed objective" box. **A `fl_union_bd_sub`** =
concat(fl_bookdemand, submodular_fl), dedup+effective_size
downstream. **B `fl_gen_rank`** = submodular_fl generates the pool,
book-demand reranks emission (bd>0 desc, bd==0 backfill in submodular
order). Arms commits `2e9b657`/`9d79ea9`/`afb560a` (full
spec+quality review; B self-weakening confirmed safe-by-construction
in the bd==0 sparse regime). 6-arm × 4-topic (`bob_*`, baselines
byte-reproduced).

| topic | greedy | sub_fl | fl_bd | resid | union | gen_rank |
|---|--:|--:|--:|--:|--:|--:|
| T11303 book-rich | .483 | .379 | **.655** | .621 | .414 | .379 |
| T10820 book-rich | .457 | .326 | **.609** | .565 | .457 | .326 |
| T10320 modern | .236 | .248 | .582 | **.618** | .418 | .248 |
| T10462 sparse | .375 | .500 | .125 | .375 | **.625** | **.625** |

**Finding 1 — the terminal structural insight.** `fl_gen_rank` is
*byte-identical* to `submodular_fl` on T11303/T10820/T10320 (recall,
spread, jaccard all equal) despite 354–563 gaps reranked by
book-demand (diag). The reorder was **entirely washed out by
`run_arm`'s downstream `effective_size` top-20 ranking**. ROOT CAUSE
of the whole tension: the deflation/seed stage only changes which
gaps *survive*, NOT the final selection — the regime-blind
`effective_size` ranking is the binding constraint. **No
deflation-stage strategy can be a universal winner because deflation
is not the stage that decides the metric.**

**Finding 2 — new book-sparse SOTA.** `fl_union_bd_sub` = **.625 on
T10462**, beating `submodular_fl`'s .500 (the prior sparse ceiling)
and every method in the entire investigation. Few sharp
fl_bookdemand gaps (diag: 37) unioned with manifold coverage +
effective_size ranking genuinely exceed either alone — real
best-of-both *on sparse*. But worst book-aware method on dense
(.414/.457/.418): submodular's 1000 gaps crowd fl_bookdemand's few
sharp ones out of the effective_size top-20 (predicted crowding).

**FINAL CONCLUSION (investigation closed at the deflation stage).**
The full deflation design space is mapped — mixture (A-keyed /
constant-s / intrinsic-residual) AND non-mixture (union / gen→rank).
No no-branch deflation is a universal winner, and we now know *why*:
the book-vs-manifold tradeoff is decided by the regime-blind
`effective_size` ranking in `run_arm`, a different (protected)
subsystem. The only genuinely new lever surfaced is the **ranking
stage** (make the survivor ranking book-aware/regime-adaptive), not
the deflation stage. Practical best numbers: dense → fl_bookdemand
(.655/.609/.582) or resid (.621/.565/.618, zero-knob); sparse →
**fl_union_bd_sub (.625, new SOTA)**. An updated regime gate
fl_bookdemand(dense)/fl_union_bd_sub(sparse) beats the old
fl_bookdemand/submodular_fl gate.

### METRIC CORRECTION + re-baseline (2026-05-18) — INVESTIGATION RESOLVED

**The headline metric was misnamed.** `run_arm` passed the FULL
deduped survivor set (not top-20) to `heldout_recall_spread`;
`heldout_book_recall_at_20` was full-set coverage,
rank-INSENSITIVE — it rewarded *emitting many gaps*, not surfacing
good ones. Fixed (commit `303ae02`, spec
`docs/superpowers/specs/2026-05-18-ranking-stage-book-aware-design.md`):
genuine `heldout_recall_at20` from `ranked[:20]` (new primary) +
retained `heldout_recall_fullset` (byte-anchor + breadth/quality
delta). 12-job re-baseline (`rb_*`). **Validity gate PASSED: every
A-run `heldout_recall_fullset` byte-matches the prior `bob_*`
numbers exactly** — the harness is sound, the correction left the
old computation untouched.

Genuine recall@20 (corrected primary):

| topic | greedy | sub_fl | fl_bd | resid | union | sub@bd | sub@x_bd |
|---|--:|--:|--:|--:|--:|--:|--:|
| T11303 | .000 | .000 | .103 | .103 | .000 | .103 | .034 |
| T10462 | .000 | .000 | .125 | **.250** | .000 | .000 | .000 |
| T10320 | .000 | .006 | .012 | **.042** | .000 | .024 | .024 |
| T10820 | .000 | .000 | **.130** | .109 | .000 | .022 | .000 |

**THE PRIOR DEFLATION CONCLUSIONS ABOVE ARE ARTIFACTS OF THE BROKEN
METRIC.** `greedy_hard`/`submodular_fl`/`fl_union_bd_sub` score ≈0.0
genuine recall@20 everywhere — their entire prior .38–.625 was gaps
beyond rank-20 (a gap-quantity artifact; e.g. submodular T11303
.379→.000, union T10462 .625→.000). The "submodular wins sparse",
"regime gate", "union .625 SOTA", "book-vs-manifold tension is
fundamental", and "ranking is the binding constraint" findings are
ALL void — the manifold-coverage methods were never good; they only
emitted ~800–1100 gaps so *some* low-ranked gap incidentally covered
held-out books.

**RESOLUTION (definitive, the whole investigation):**
1. There is **no book-vs-manifold tension** under the correct
   metric — it never existed; it was the misnamed metric.
2. **`fl_bookdemand_resid` is the winner** — intrinsic, zero-knob,
   no-branch; best or tied genuine recall@20 on all 4 topics
   (decisively best on book-sparse T10462 .250, where every
   coverage method = .000). Only the *few-sharp-book-anchored*
   methods (resid, fl_bookdemand) surface held-out-book regions in
   their top-ranked gaps; resid ≥ fl_bookdemand.
3. Book-aware *ranking* of a book-blind pool cannot rescue it
   (`sub@effsize_x_bd` ≈.0–.034, often < pure `@bookdemand` — the
   multiplicative effsize factor dilutes). The survivor set must be
   book-anchored *at generation*; ranking is NOT the lever. The
   simplest intrinsic book-anchored deflation already wins.
4. Absolute recall@20 is low (.10–.25 best) — expected for an
   honest top-20-vs-held-out-books proxy; the RELATIVE ordering
   (resid/fl_bd ≫ all coverage methods) is the robust signal.
   Triangulate with qualitative-primary as always.

**Status: INVESTIGATION RESOLVED. Winner = `fl_bookdemand_resid`
(intrinsic, zero-knob, no branch). No tension, no gate, no
best-of-both needed — the simplest book-anchored method wins once
the metric is correct. The single most instructive artifact is the
metric-misnomer diagnosis itself. Ready to migrate
→ docs/gap_deflation_results.md (winner=resid + the metric-flaw
lesson) and transition the refuted coverage/regime/union/ranking
detours to dead_ends. Pending final sign-off.**
