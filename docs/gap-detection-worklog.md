# Gap detection — living worklog

*Status-tracked index of all work on the fg/bg gap-detection front. Update discipline: items
move Proposed → In progress (on dispatch) → Completed (with a one-line numeric verdict +
artifact pointer) or → Rejected (with the refuting number). Keep entries terse; depth lives in
the linked handoffs/reports. Maintained by the working agent(s); owner edits freely.*

Related: [2026-06-11 plant-and-recover handoff](handoffs/2026-06-11-gap-plant-recover.md) ·
"Gap detection, revisited" in [curriculum-applications.md](backlog/curriculum-applications.md) ·
code `artifacts_scripts/gap_detection/` · results `artifacts/gap_detection_eval/` ·
**one-figure summary** `artifacts/gap_detection_eval/figs/method_map.png` (lab heatmap × wild
forest; regenerate via `plot_method_map.py`, reads banked results.json)

## Completed

- **A. Radial score-fusion** (2026-06-11, follow-up subagent; same 16 trials, identity
  asserted; 25 tests green) — SCOPED: fusion works and needs no oracle — rank-real 6.5
  (baseline 10.5), rank-all 13 with probes eligible, AUC 0.834 (vs 0.842), probes in
  top-50: 50→6.6, P@50 0.28; dominates the full statistic (386 / 0.78 / 0.00). UNSCOPED:
  veto can NOT replace the scope quantile — best variant (Δ≤log2) median rank 8,708 ≈ full
  statistic 8,860 vs scoped 13; only ~5% of off-field bg exceeds Δ=log3 in 128-d LSA and
  the loose unscoped hull leaks ~5% of probes (24/50 remain in top-50). Verdict: veto
  COMPLEMENTS scope (in-scope guard), habitability's load-bearing form stays the candidate
  universe. Code `fusion.py`; rows `results.json:task_a`; report §"Follow-up Task A".
- **B. Empirical-null calibration** (2026-06-11, same dispatch) — NEGATIVE on the ordering
  question, cleanly measured: stratified-by-r_B empirical p edges the parametric LCB
  in-sample (AUC 0.825 vs 0.817, rank 354 vs 422; AUC wins 11/16) but the split-half
  held-out-null check inverts it (0.829 vs 0.842) — the gain is the null grading its own
  calibration; pooled p is order-identical to raw Δ (sanity ✓). Keep the LCB. POSITIVE as
  A's veto: Tukey-fence ceiling self-tunes to ~log 2 and matches the best fixed ceiling
  unscoped (8,656 ≈ 8,708) — removes the C hyperparameter; in-scope it over-vetoes
  (AUC 0.79), so scoped fusion keeps the fixed gate. Pooling strategy + sensitivity
  documented in report §"Follow-up Task B". Code `calibration.py`;
  rows `results.json:task_b`.
- **C. Bipartite-Gramian geometry test** (2026-06-11, same dispatch) — NEGATIVE, decisively:
  concept gaps do NOT become radial in the doc×keyword walk geometry (full-statistic AUC vs
  matched controls 0.47 LSA → 0.49 walk; all sensitivity configs 0.46–0.53; median orphaned
  supporters still 0). The seductive best-rank gain (188→28) is a generic pattern-supporter
  effect — matched CONTROL supporters rank at median 3.5, higher than the planted region.
  Radial detection collapses in walk geometry (r_F AUC 0.84→0.51): the two geometries are
  near-orthogonal; neither subsumes the other. Mechanism is real though — the unit fixture
  (rare shared keywords) gives walk AUC 1.0 where raw counts see 0.52; wiki concepts are
  co-occurrence patterns over COMMON top-2000-df keywords whose walk mass dilutes. Miner's
  monopoly STANDS; species↔detector matching is fundamental at this corpus/keyword budget.
  Possible flip left on the table: a low-df-tail keyword space. Code `walk_geometry.py`;
  rows `results.json:task_c`; report §"Follow-up Task C".
- **P1. Miner cross-species sweep** (2026-06-11, polish dispatch; identity asserted, banked
  voltage rank 32 + miner rank 436 reproduced) — null prediction PARTIALLY REFUTED on radial,
  corridor-conjunction hypothesis REFUTED: radial removals DO leave minable signatures
  (15/16 found at ≥80% wipe, median attributable rank 217; 16/16 at ≥50%, rank 30.5; doc-layer
  AUC 0.67) but far below the home detector (doc rank 317.5 vs fused 6.5, AUC 0.67 vs 0.83);
  corridor finds only 4/16 at ≥80% (median rank 1,843, half cross-cluster conjunctions —
  the signature exists but bg support ~10-30 keeps z low; doc rank 423.5 vs voltage 12);
  concept anchor exact (16/16 @ 128.5). → Track 1 implication: miner-led composite REJECTED —
  real fusion of species-matched detectors required; miner radial-attribution usable only as
  cheap corroboration. Code `polish.py`+`run_polish.py`; rows `results.json:task_p1`;
  report §"Follow-up Task P1".
- **P2. Level-set residualization** (2026-06-11, polish dispatch; pockets + all 48 banked
  level-set baselines re-asserted) — differencing (after − intact-fg baseline) REVIVES the
  component machinery as a change detector: recall@top-3 0.00 → 0.50 radial / 0.31 corridor
  at the banked 5% selection, and the load-bearing fix is matching selection to change scale —
  at top-frac 0.01: radial 1.00 (median IoU 0.40), corridor 0.75, planted component rank 1
  when found; concept stays 0.00 (no geometric change exists — miner keeps it). Natural
  pockets (abs ranks 1–2) vanish from the component list in 63–75% of trials, median 4–20 of
  151/91 members surviving selection. Caveat on record: radial/corridor cores are defined by
  the same before/after r_F growth (near-oracle); non-circular content = budget finding +
  concept negative + pocket cancellation. → Track 1/2 implication: NON-viable for Track 1
  single-snapshot candidate generation (needs the before-field oracle); VALIDATED verbatim as
  Track 2's T vs T+Δ snapshot differencer. Rows `results.json:task_p2`; report §"Task P2".
- **P3. Voltage all-pairs + resistance metric** (2026-06-11, polish dispatch; banked voltage
  rank 32 + concept LSA rank 1154 reproduced; all 66 qualifying pairs, planted coverage 16/16) —
  (a) SPECIES-AGNOSTIC CHANGE FIELD FOUND (the dispatch's surprise): max-over-pairs
  |flow-share change| on the joint fg+bg kNN graph localizes ALL THREE species with no
  species/pair knowledge — radial median rank 1.5 / AUC 0.79 (home fused 6.5 / 0.83), corridor
  3.5 / 0.82 (BEATS the single-pair voltage specialist, 12 / 0.69, without the pair oracle),
  concept 20.5 / 0.64 (node removal perturbs flow even with zero density dent). After-only and
  static variants sit at chance (AUC 0.51–0.55): the CHANGE carries all signal. Physics note
  measured in: removing a bridge LOWERS total current and the corridor's flow share (the fg
  superhighway stops pulling flow), so share normalization + |Δ| are required. 3-NN-adjacency
  subset within 0.03 AUC of all-66 (aggregation robust). (b) Commute/resistance geometry
  NEGATIVE: concept AUC 0.48 / 0.47 / 0.49 (bipartite d=128 / d=64 / LSA-graph commute; refs
  LSA 0.47, walk 0.49), orphans 0 — the ALL-paths metric fails exactly where one-hop failed;
  flat bipartite spectrum (λ ∈ [0.35, 0.77]) = the commute-degeneracy regime. Radial:
  keyword-commute collapses (r_F AUC 0.52); LSA-graph commute keeps AUC 0.79 but rank 10 → 291.
  → Track 1/2 implication: voltage DOES generalize to the composite's geometric half but in
  Track 2 FORM ONLY (the T vs T+Δ change field is the geometric workhorse, pairing with P2's
  residual components); Track 1 single-snapshot suite unchanged; commute earns no place.
  Rows `results.json:task_p3a/task_p3b`; report §"Task P3a"/"Task P3b". Implementation note:
  shift-invert eigsh on the 32k bipartite Laplacian explodes (hub fill, 87 CPU-min, aborted) —
  `polish.commute_embedding` uses matvec-only ARPACK on N = D^-1/2 W D^-1/2 + Lovász identity.
- **P4. Low-df miner lattice** (2026-06-11, polish dispatch; tf-idf rebuild identity + all 16
  banked planted ranks re-asserted) — decisive DOUBLE NEGATIVE: planted-combo median rank
  128.5 → 128.5 under every augmentation (cap-10k tail df∈[5,31] / full 12,316-term df∈[5,50]
  window → 2,111 / 5,477 tail itemsets at tier min_support 10; min_len 1 worsens to 227);
  best-attributable tail share 0.00. Mechanism: among fg-absent itemsets, Fightin'-Words z is
  monotone in bg support — the ~128 itemsets above a plant are the corpus's NATURAL emerging
  patterns (standing gaps), so the champion's bound is corpus structure, not lattice
  resolution. C's noted "possible flip" REFUTED: walk geometry at chance in augmented (0.48)
  and tail-only (0.45; 2,368 isolated docs) spaces — planted wiki concepts share common-keyword
  COMBOS while members' rare vocabularies are heterogeneous; the tail has nothing to bind.
  → Track 1/2 implication: champion boost = ZERO — keep the miner exactly as is (top-2000-df,
  scoped, min_support 30); genuinely-new-rare-term (emerging-topic) gaps remain Track 2's
  regime via P2/P3a differencing. Rows `results.json:task_p4`; report §"Task P4".
- **Plant-and-recover validation v1** (2026-06-11, subagent; 18 tests green) — synthetic
  ground-truth gaps (radial / concept / corridor × 16 seeded trials) planted in a 30k wiki
  subsample, matched hypothetical-removal controls. Headline verdicts:
  - Species↔detector matching CONFIRMED: radial → kNN/r_F (AUC 0.84 best form); concept →
    scoped emerging-pattern miner (16/16, median rank 128/36,199 = top 0.4%); corridor →
    voltage bridging w/ extended electrodes + `4φ(1−φ)` (median best rank 12).
  - Habitability is load-bearing as the candidate UNIVERSE (hard scope ~23×: 8,860→386),
    NOT as a soft multiplicative weight (g^a hurts within scope: AUC 0.84 vs 0.78).
  - LCB variance term: mechanism real, effect marginal (0.74→0.78); keep, don't rely on.
  - Full details + honest-ceiling notes (backfill bounds detectability): see handoff.
- **Track 1 finish: type-blind composite + sensitivity frontier** (dispatched 2026-06-11,
  owner "go after any and all leads"; **COMPLETED 2026-06-11, both parts, 56 tests green**
  incl. 4 new contracts: percentile-fusion, noisy-OR/abstain, pair-emptiness fixture,
  frontier footprint-monotonicity) — the per-type asterisk "the gap's type was known" is
  REMOVED and priced. Code `composite.py` + `run_t1_composite.py`; rows
  `results.json:task_t1_comp`; report §"Track 1 finish, Part A"/"Part B".
  - **Part A — type-blind composite benchmark: blindness is CHEAP on radial, EXPENSIVE on
    corridor, MOOT on concept** (48 mixed trials = the banked plants in a seeded
    balanced-deck shuffle; identity asserted, banked ranks reproduced per species incl.
    all 16 corridor voltage ranks). Composite = geo (fused r_F+veto) + pat (miner doc
    layer) + volt (single-pair voltage on 3 SELF-selected pairs by band-emptiness,
    after-fg KMeans-12); arbitration = MAX of in-trial channel percentiles (noisy-OR, no
    channel leads — P1 respected; gated routing rejected on measurement: emptiness
    false-fires on radial at the same 0.60–0.64 level as true corridors). COST OF
    BLINDNESS (oracle → composite; median rank / mean AUC / found@50): radial
    6.5/0.83/0.75 → 12.5/0.78/0.62 (+6 ranks); corridor 12/0.82/0.69 → 67/0.78/0.50
    (+55; decomposed: pair-blindness 12→29.5 at region coverage 0.98, arbitration
    interleave 29.5→67); concept 40.5/0.56/0.62 → 37.5/0.52/0.56 (−3 — the doc-space home
    is already near-chance AUC; the miner's real power is itemset-space 128.5/36,199,
    inexpressible in a doc-ranked list). Overall blind found@50 0.56, median rank
    33.5/7,139; probes 6.5/50 (= geo's known leak, unworsened). Ablation: MAX ≻ mean
    (88.5 rank, found 0.35 — dilution confirmed), ≈ RRF (38.5/0.56), ≈ geo+volt.
  - **Part B — sensitivity frontier: the minimum recoverable gap is ~64 docs (~4% of the
    field)** (sizes 8–256 × k {5,10,20} PAIRED at the 16 banked centers, banked removed
    sets reproduced exactly; banked-budget anchor: size-128 row k=10 rank 6 / AUC 0.86 ≈
    banked 6.5/0.83; two designs run + rejected on record — per-cell resampling = location
    lottery, random-center pairing shows ≤32-doc removals leave no witnessed footprint at
    typical locations). Frontier (k=10 rank/AUC/found@50 by size): 8 = no core anywhere;
    16 = 4/16 constructed, 4418/0.60/0.00; 32 = 10/16, 1558/0.83/0.20 (causal AUC, no
    production-visible recovery); **64 = 13/16, 30/0.87/0.54**; 128 = 16/16, 6/0.86/0.94;
    256 = 16/16, 1/0.89/0.94. k=5 ≈ k=10; k=20 over-smooths (threshold doubles to 128).
    Corridor (oracle-pair voltage): 8/16-doc = no core; 32 = 4/16; **64 = 13/16,
    42/0.87/0.54**; 128 = 16/16, 14/0.82/0.69. SPEC (production claims quote this): fused
    r_F+veto k=5–10 recovers 64-doc radial pockets (AUC 0.85–0.87, found@50 0.54–0.62) and
    oracle-pair voltage 64-doc corridor bands (0.87, 0.54); below 64 the orphan-core
    footprint itself collapses.

## Completed (Track 2)

- **Track 2, wiki phase: temporal holdout validation** (dispatched 2026-06-11, owner "SGTM";
  COMPLETED 2026-06-11, 44 tests green) — VERDICT: gate passed only after a forced T-slide
  (store = 400k oldest ids, 2001–2004; dispatched 2014–2022 rows = 0 arrivals); pre-committed
  T=2003-01-01/Δ=2y (63k in-scope arrivals); proxy p90 23d (boundary 24.6d). With the
  matching machinery calibrated on random-location nulls (v1 full-disjointness biased the
  null to 1.27 — banked; final caliper matcher null 1.04 [0.86, 1.22]) and near-duplicate
  locations handled by a clustered bootstrap, ONE predictor carries real-arrival signal:
  absolute LEVEL-SET components, fill ratio 1.28, clustered CI [1.06, 1.48] (14 groups) —
  the Track-1-refuted discovery detector is the one whose structures actually fill. Fused
  1.10 [0.89, 1.36] and voltage 1.80 [0.85, 2.99] = honest negatives; miner's naive 1.26
  [1.14, 1.39] dissolves under clustering (4 distinct locations; top patterns are generic
  register pairs) = inconclusive + needs diversity control. Outcome instruments mutually
  disjoint (Jaccard ≤ 0.005) and uncorrelated with predictors (P@50 ≤ chance 0.063/0.052
  except voltage-vs-P3a 0.48 = self-correlation ρ=0.61): the 2-year regime is explosive
  growth (arrivals ≈ 8.5× corpus), not the marginal-change regime of the plant validation.
  fg re-find FAILED at T (field fragmentation; continuity rule on record). Code
  `wiki_temporal.py`/`t_corpus.py`/`t2_outcomes.py`/`run_t2_*.py`; rows
  `results.json:task_t2_wiki`; report §"Track 2 — wiki temporal holdout"; caches
  `cache/wiki/creation_dates/` (2,840 exact timestamps) + `cache/t2_*`. Phase log below.
  - [x] **Phase 0 — arrival-density gate: PASSED, with a forced T-range slide** (2026-06-11).
    Store discovery: `wiki_400k` is a bounded cirrus-dump slice = the 400k OLDEST surviving
    page_ids (12..1,348,640; file order perfectly id-sorted), creation span 2001-08-28..
    2004-12-31 (240 stratified exact API timestamps; thresholds 2003-01-01≈id 163.5k,
    2004-01-01≈id 417k). The dispatched candidate years 2014–2022 therefore have ZERO
    arrivals (gate rows recorded as 0) — but the design generalizes by sliding T into the
    store's own era. Density table (rough proxy + banked-LSA embed, identity err 1.2e-07;
    scope = q0.25 rule from ≤T members of the banked subsample): T=2002-01: pre-T-eligible
    16,387 / 2y-arrivals 125,163 / in-scope 18,050 / fg-at-T 138; 2002-07: 26,461/200,261/
    26,544/205; **2003-01: 70,626/262,127/63,151/320**; 2003-07: 99,527/233,226/52,777/512;
    2004-01: 141,550/191,203/42,935/711; 2004-07: 226,722/106,031/25,278/1,049.
    **PRE-COMMITTED (counts only, no outcomes computed): T=2003-01-01, Δ=2y** — rule:
    maximize in-scope arrivals subject to pre-T eligible ≥ 32k (size-matched 30k T-corpus
    feasibility); Δ=2y and 4y identical at this T (store ceiling 2004-12-31 truncates;
    2y label honest at 730/731-day coverage). Growth-control window (2001-01, 2003-01]
    missing its first 7 months (earliest surviving first-rev 2001-08-28) — on record.
    Choice + table: `artifacts/gap_detection_eval/cache/t2_choice.json`, `t2_gate_table.json`.
  - [x] **Phase 1 — creation dates: proxy VALIDATED** (2026-06-11). 2,839 exact API
    first-revision timestamps (240 rough-stratified + 2,000 stratified + 600 boundary-band
    ±40k ids around id 163,531; concurrency 16, all responses cached to
    `cache/wiki/creation_dates/`, 1 missing / 2 redirects / 0 errors). Isotonic id→date fit
    on the even half, validated on the odd half (n=1,419): median |err| 0.03 days, p90 23.1
    days; NEAR-BOUNDARY (±1y of T) p90 24.6 days — far under the 180-day gate, no further
    densification needed. Wrinkle rates (held-out |err|, the redirect-turned-article /
    history-merge / undeletion tail): >90d 1.7%, >180d 1.2%, >365d 0.42%. Final proxy refit
    on all 2,838 points; exact dates override proxy where known. Dates assigned to all 400k
    store rows + the 30k banked subsample: `t2_dates.npz`, `t2_calibration_report.json`.
  - [x] **Phase 2 — leakage-free T-corpus: BUILT; fg re-find FAILED, continuity rule applied
    (deviation on record)** (2026-06-11). T-corpus = uniform 30k sample of the 70,608-doc
    pre-T eligible pool (banked pre-T docs force-included: 5,993 of 6,389 survive the
    nnz/trim replay); TF-IDF (18,445 terms), LSA-128 (EVR 0.183), top-2000-df keywords,
    MiniBatchKMeans-40 — ALL fit on ≤T docs only (leakage guard unit-pinned; post-T-token
    fixture). SANITY FINDING: the seeded procedure does NOT re-find the society/law/politics
    field at T — its coherence-max pick is a generic-terminology cluster (banked-fg recall
    0.075) because the plant-era interleave≥5 gate excludes every topical cluster; the
    banked fg FRAGMENTS across T-clusters (academia/philosophy/economics 86, religion 62,
    law 52, economy 36 of 292 present members). Continuity rule (deterministic, recorded):
    fg = max-banked-overlap size-qualified cluster = academia/philosophy/economics
    (n=1,012; terms: sciences, professor, academic, philosophy, nobel, economics; banked
    recall 0.295). Scope (T geometry): thr 1.063, 7,247 scoped bg of 28,988 (banked scale:
    7,139/28,554). Frozen-transform embeddings: 255,683 arrivals in (T, T+2y],
    70,607 growth-window (T−2y, T] pages. Approximations on record: 2026 text for pre-T
    docs (drift), deletion survivorship, proxy dates (boundary p90 24.6d), store-ceiling
    1-day truncation. Artifacts: `t2_corpus(.npz/_K.npz)`, `t2_arrivals(.npz/_K.npz)`,
    `t2_growth.npz`, `t2_models.joblib`, `t2_phase2_meta.json`.
  - [x] **Phase 3 — predict, score, compare: DONE; one positive, three honest negatives**
    (2026-06-11). Pre-committed K=50/M=20/B=2000; radius = 10-NN scale vs scoped bg;
    covariates log r_B10 / log r_F10 / log1p growth(T−2y,T]. Machinery hardening (calibrated
    on random-location nulls ONLY, never detector outcomes; both passes banked): (i) v1
    full-disjointness d>r_f+r_c annihilates the sparse tail on the unit sphere → random
    null 1.27 [1.02, 1.56], density SMD up to 1.72 → replaced by calipers
    (0.2/0.5/0.5σ, progressive) + center-disjointness d>max(r_f,r_c) → null 1.04
    [0.86, 1.22], balance ≤ 0.07σ; (ii) near-duplicate locations (miner top-50 collapses to
    4 neighborhoods; its top patterns are register pairs {use, used}…) → clustered
    bootstrap over greedy location groups. RESULTS (fill ratio, clustered CI): fused 1.10
    [0.89, 1.36] honest negative; miner 1.26 [0.89, 1.31] inconclusive (4 groups);
    voltage 1.80 [0.85, 2.99] honest negative; **levelset 1.28 [1.06, 1.48] POSITIVE**
    (14 groups; caveat: 50/50 caliper-relaxed, residual balance 0.11/0.21/0.25σ);
    random50 1.04 [0.86, 1.22] ✓. Instruments: P2 (2 components, 67 docs, fg+11,605
    arrivals) vs P3a (61 pairs, 73 peaks) Jaccard 0.0; P@50 ≤ chance (0.063/0.052) for all
    predictors except voltage-vs-P3a 0.48 = self-correlation (ρ=0.61, same
    graph/electrodes). Realized-change top components: medical/disease cluster
    (Hyperthyroidism, Pneumonia, …) and literature/biography mix. Rows
    `results.json:task_t2_wiki.phase3` (+`phase3_v1_sum_disjointness`); report section
    rendered by `run_t2_report.py`.
  - **Descriptive pass (2026-06-11)**: the levelset positive is heavy-tailed + confounded.
    8 flagged components are coherent topics (math/medicine/networking/cinema/physics/
    economies/SF-lit/myth); the 1.28 ratio is carried by cinema (fill 148, 2.2× pool 68) and
    SF-lit (103, 1.5×) — categories editors MASS-PRODUCED in 2003–04 (filmographies,
    bibliographies), i.e. editorial productivity, not gap-closing; the growth match doesn't
    control for "category under active expansion". The most gap-like component (large math
    neighborhood) fills at baseline (70 vs 68) yet eats the whole top-50 budget. Verdict: the
    pipeline detects high-subsequent-ACTIVITY regions; on wiki that ≈ pop-culture completion.
    Strengthens the OA case (citation "filling" has no editorial-mass-production analogue).
    Code `describe_t2_levelset.py`; write-up `artifacts/gap_detection_eval/t2_levelset_descriptive.md`.

- **Track 2, OA phase: NLP temporal holdout** (dispatched 2026-06-11, owner "LGTM"; **COMPLETED
  2026-06-11, all 4 phases passed, 50 tests green**) — VERDICT: the wiki finding REPLICATES
  CLEANER in the OA marginal-growth regime and the leak-free citation graph adds an orthogonal
  win. Gate PASSED on the locked universe (T10181, 101,976 works 2008–2019; **T=2015-01-01,
  Δ=4y**, 6,263 in-scope arrivals ≈ 0.19× pre-T pool — no editorial-mass-production analogue).
  fg = DISCOVERED dependency/syntactic PARSING cluster (after a language fix dropped a
  degenerate French-stopword island). With the matcher calibrated on random-location nulls
  FIRST (**random50 0.92 [0.75, 1.08]** — no wiki-v1 inflation): **absolute LEVEL-SET components
  are again the ONE predictor that out-fills growth-matched controls — 1.92 clustered CI
  [1.63, 2.23]** (19 groups) — and here they COINCIDE with the realized-change instrument (P@50
  vs P2 0.98) and the fill is real parsing-adjacent arrivals (tagging/MT/segmentation), not
  pop-culture mass-production. Scoped fusion an honest negative (0.93); miner dissolves under
  clustering (1.23, 2 groups) but its fg-absent patterns are the face-valid **MT gap**
  ({machine, translation}). THE OA WIN as a DISSOCIATION: the dated citation graph densifies
  14,002 → 33,268 edges over (T,T+Δ]; P3a citation-flow-change + voltage_cite localize the
  **CoNLL/treebank-benchmark** cluster (self-agree P@50 0.74) but those nodes UNDER-fill with
  new abstracted papers (voltage_cite 0.71 INVERTED) — where citations reroute ≠ where papers
  land. Locked design held: universe T10181; counts-pre-committed T/Δ; fg discovered; fallback
  (T10181+T10028+T10201) NOT needed. The OA-specific win: dated citation graph (edge present at
  T iff citing.date ≤ T → exact T-graph for P3a). Rows `results.json:task_t2_oa`; report
  §"Track 2 — OA NLP temporal holdout"; descriptive `artifacts/gap_detection_eval/t2_oa_descriptive.md`;
  caches `cache/oa/nlp_t10181/` (101,976 works) + `cache/t2_oa_*`; spec
  `docs/handoffs/2026-06-11-oa-nlp-temporal-brief.md`. Code mirrors the wiki path: `oa_corpus.py`
  (data layer, swaps the wiki store) + `run_t2_oa_phase{01,2,3,report}.py` + `describe_t2_oa.py`,
  reusing `t_corpus.py`/`t2_outcomes.py`/`detectors.py`/`fusion.py`/`polish.py` unchanged.
  Phase log below.
  - [x] **Phase 0 — acquisition + arrival-density gate: PASSED on the LOCKED universe**
    (2026-06-11). Cursor-paged T10181 pub-2008–2019: **101,976 works** cached to
    `cache/oa/nlp_t10181/` (510 pages, polite pool, all dated). Abstract coverage 64.4% any /
    57.4% ≥200ch / 53.9% ≥200ch+English (matches brief ~66%; English-only after the language
    fix — see Phase 2). At **T=2015-01-01, Δ=4y** (English-filtered): pre-T eligible 32,384,
    arrivals (T,T+Δ] 18,721 eligible / **6,263 in-scope** (≫150 gate — NO fallback). Marginal
    regime: in-scope arrivals ≈ 0.19× pre-T pool (vs wiki's explosive 8.5×) — kills the
    editorial-mass-production confound. Provisional fg cluster size 1,131 (in 1–3k).
    Intra-universe citation edges 246,774 (51,885 works carry ≥1 in-universe ref).
    Table+choice: `cache/t2_oa_gate_table.json`, `t2_oa_choice.json`; banked
    `results.json:task_t2_oa.phase0_gate`.
  - [x] **Phase 1 — dates + dated citation graph: DONE** (2026-06-11). publication_date is
    EXACT (no proxy — simpler than wiki). Dated citation graph (edge present at T iff
    citing.date ≤ T → exact T-graph, no future leakage; toy-fixture leakage guard green):
    **T-graph** 58,635 dated nodes / 106,284 edges / 26,070 non-isolated / density 3.09e-05 /
    max in-deg 448; **T+Δ-graph** 95,554 / 223,989 / 43,219 / 2.45e-05 / max in-deg 973 — the
    citation structure ~doubles over (T,T+Δ], the leak-free object wiki's undated links
    couldn't give. COO persisted to `cache/t2_oa_graph.npz` (Phase 2); stats in
    `phase0_gate.graph_T{,_plus_delta}`.
  - [x] **Phase 2 — leakage-free T-corpus: BUILT; fg = genuine PARSING cluster** (2026-06-11).
    Decoded abstract_inverted_index; refit TF-IDF (vocab 10,166) / LSA-128 (EVR 0.167) /
    KMeans-40 / scope ALL on ≤T English docs only (frozen-transform identity + leakage-guard
    toy fixtures green). LANGUAGE FIX (deviation on record): the first build's coherence-max fg
    was a degenerate FRENCH-stopword island (terms des/la/les…, coherence 0.96, bg-interleave
    0) — ~5% of T10181 abstracts are non-English; with stop_words=english their function words
    cluster perfectly. Added an `is_english` filter (English vs non-English stopword mass);
    3,527 non-English ≥200ch works dropped universe-wide (missingness model = drop, like the
    char rule). Rebuilt fg = **DISCOVERED cluster: dependency/syntactic PARSING** (terms:
    parsing, dependency, parser, treebank, syntactic, trees, parse…; n=1,041, coherence 0.59,
    bg-interleave 1.0 — interleaved, nameable, a real post-2015-active NLP subfield). T-corpus
    n=32,079; scope 7,760 of 31,038 bg; frozen embeddings 18,505 arrivals / 18,139 growth.
    Caches `t2_oa_corpus(.npz/_K.npz)`, `t2_oa_models.joblib`, `t2_oa_arrivals(.npz/_K.npz)`,
    `t2_oa_growth.npz`, `t2_oa_graph.npz`, `t2_oa_phase2_meta.json`; banked phase2_t_corpus.
  - [x] **Phase 3 — predict / score / compare: DONE; levelset POSITIVE (cleaner than wiki),
    the MT gap is face-valid** (2026-06-11). Pre-committed K=50/M=20/B=2000; radius=10-NN scale
    vs scoped bg; covariates log r_B10 / log r_F10 / log1p growth(T-Δ,T]; reused the banked
    caliper + center-disjoint matcher verbatim. Machinery null calibrated FIRST:
    **random50 0.92 [0.75, 1.08]** (50 groups, spans 1, balance ≤0.06σ) — clean, no wiki-v1
    inflation (toy-fixture matched-null test pins ≈1.0). RESULTS (fill ratio, clustered CI):
    **levelset 1.92 [1.63, 2.23] POSITIVE** (19 groups; balance 0.16/0.13/0.19σ) — and unlike
    wiki it COINCIDES with the realized-change instrument (P@50 vs P2 = 0.98 vs chance 0.15);
    fused 0.93 [0.77, 1.13] honest negative (the emptiest in-scope points fill at the matched
    rate); miner 1.23 [0.86, 1.24] inconclusive (collapses to 2 groups under clustering, like
    wiki); voltage_emb 1.05 [0.85, 1.30] honest negative; **voltage_cite 0.71 [0.57, 0.88]
    INVERTED** (citation-corridor nodes UNDER-fill with abstracted arrivals though P@50 vs P3a
    = 0.74 = self-correlation on the citation graph). THE OA WIN works: P3a flow-change on the
    DATED CITATION GRAPH difference — T-graph 14,002 edges → T+Δ-graph 33,268 edges as 6,137
    in-scope arrival conduits attach (leak-free; the buggy first pass had T=T+Δ=13,731 because
    arrival-originated edges were dropped — fixed by augmenting the node set with arrival
    conduits). FACE-VALIDITY: the miner's top fg-absent patterns are **{machine, translation},
    {statistical, translation}, {machine, statistical, translation}** — i.e. **machine
    translation / NMT**, the biggest post-2015 NLP growth area, absent from the parsing fg.
    Rows `results.json:task_t2_oa.phase3`; report §"Track 2 — OA NLP temporal holdout".
    Descriptive (keyword labels of flagged regions) `describe_t2_oa.py`.

- **Reroute-vs-arrival dissociation (task_oa_dissoc)** (dispatched 2026-06-11, born from
  task_t2_oa's voltage_cite inversion 0.71 [0.57, 0.88] + P3a self-agreement P@50 0.74;
  **COMPLETED 2026-06-11, all 3 questions answered, 52 tests green**) — VERDICT: the
  inversion is INFRASTRUCTURE, not error, and the two fields are two different instruments.
  (Q1) The under-filling flow hubs are canonical-reference papers — voltage_cite top-50 =
  **54% infrastructure** under the pre-specified operationalization vs 14.6% cov-matched
  (+0.40 [0.27, 0.54]) and 31.2% degree-at-T-matched (+0.165 [0.04, 0.30]) controls;
  face-valid CoNLL shared tasks / Universal POS Tagset / treebank-annotation papers.
  (Q2) Leading-indicator REFUTED: W1=(2015,2017] reroute peaks do NOT predict W2=(2017,2019]
  arrivals above growth-matched nulls — strict 4-cov primary **1.08 [0.81, 1.33]** (nulls
  calibrated ≈1 first); the W1 field already ≈ the full-Δ field (ρ=0.91) — hubs persist,
  there is nothing temporal to lead with. (Q3) The fields are ORTHOGONAL: Spearman −0.10
  raw / +0.01 covariate-residualized, top-50 Jaccard 0.0, HH 36 vs ~49 independence.
  **The two fields, formalized:** the citation FLOW-CHANGE field (P3a: max-over-pairs
  |Δ flow-share| on the dated citation graph, T vs T+Δ) measures REDISTRIBUTION OF
  REFERENCE TRAFFIC over the existing ≤T literature — high where a node's role in the
  citation circuit between subfield cores changed; empirically the canonical-reference
  stratum (benchmarks, tagsets, treebanks, newly-canonized methods): what the field now
  ROUTES THROUGH. The ARRIVAL-DENSITY field (abstracted arrivals in the own-radius
  neighborhood, frozen T-embedding) measures TOPICAL OCCUPATION — where new papers' content
  lands; its connected sparse structures (level-sets) are the validated arrival predictor
  (1.92). Papers CITE infrastructure but WRITE about topics, so the fields dissociate by
  mechanism and neither leads the other; flow-change is the right tool for consolidation /
  canon formation (benchmark adoption, instrument identification, curriculum backbone),
  arrival-density for where new work lands (topic-space gap filling). 2x2 regimes: HH =
  revolution front (attention-NMT founding papers), HL = static canon (Q1's hubs), LH =
  production zones (WMT/QA arrivals without rewiring), LL = background. Code `oa_dissoc.py`;
  rows `results.json:task_oa_dissoc`; report §"Follow-up task_oa_dissoc"; titles cache
  `cache/oa/titles/` (589 ids, 12 polite-pool calls — display_name absent from the phase-0
  select list; justification on record). Question log below.
  - [x] **Q1 under-filling hub composition: INFRASTRUCTURE hypothesis SUPPORTED**
    (2026-06-11; identity asserts green — vc50 fill 5.5/42 groups, P3a 3 pairs/78 peaks/
    14,002→33,268 edges, P@50 0.74 all reproduced; titles fetched 589 ids / 12 calls,
    cached `cache/oa/titles/`). Operationalization (pre-specified, pool quantiles):
    g = dated-graph in-degree growth over (T,T+Δ], a = own-radius arrival fill,
    IR = log1p(g)−log1p(a); INFRA := g≥q75(citation-active pool)=4 AND a≤median=9.
    Composition: **voltage_cite top-50 = 54% infrastructure** vs 14.6% cov-matched
    controls (diff +0.40, clustered CI [0.27, 0.54]) and 31.2% degree-at-T-matched
    (+0.165 [0.04, 0.30]); P3a peaks 34.6% (+0.24 [0.13, 0.35] vs cov; binary n.s. vs
    degree-matched +0.05 [−0.05, 0.15] but continuous IR stays above, +0.85 [0.49, 1.31]).
    Hubs are OLD + CITED + EMPTY: median deg_T 18 vs pool 1, age 4.6y vs 3.2, fill 4.5 vs 9,
    cited_by_2026 37 vs 4. Top hubs face-valid CoNLL/UD-style: CoNLL-2008/2009 shared
    tasks, Universal POS Tagset (g=219, fill=3), treebank/annotation-scheme papers; title-cue
    rate (secondary) 30%/28% vs 14%/19% controls, levelset 2%. Caveat on record: the binary
    rule also catches canonical METHOD papers (Zettlemoyer-Collins g=105 fill=0) — "infra"
    here = canonical-reference papers, instruments + canon; cue-titled half is datasets/tasks
    proper. Levelset contrast: IR median −0.54 (arrival-dominated), growth spread n_eff 29.5
    vs vc50's 18.1. Rows `task_oa_dissoc.q1`; report §"task_oa_dissoc Q1".
  - [x] **Q2 leading-indicator hypothesis REFUTED — reroute and arrival are different
    phenomena** (2026-06-11; window-split leakage guard unit-pinned: W1 field uses only
    ≤2017 citing acts + (2015,2017]-dated conduits). Nulls calibrated FIRST, all span 1
    (4-cov seeds 17/18/19: 0.99 [0.85,1.15] / 1.12 [0.93,1.30] / 1.11 [0.83,1.39]; 3-cov
    W2 0.89 [0.72,1.05]; W1 0.96 [0.76,1.14]). PRIMARY (W1-reroute top-50 → W2 arrivals,
    strict 4-cov matching incl. log1p W1-arrival-fill): **1.08, clustered CI [0.81, 1.33]
    — CI spans 1**; 3-cov 1.18 [0.76, 1.73]; top-1% peaks sens 1.01 [0.82, 1.20];
    same-window reference 1.15 [0.84, 1.52]. The W1 (2y) field is STABLE vs the full-Δ
    field (Spearman 0.91, P@50 vs full peaks 0.86): reroute hubs persist, they don't march
    ahead of arrivals. Note the asymmetry vs phase-3: the static-share predictor
    (voltage_cite) UNDER-fills (0.71) while the change-field top-50 fills at the matched
    rate — the inversion belongs to the old high-share backbone, not to change peaks. No
    early-warning signal; the two fields measure different things (→ Q3). Rows
    `task_oa_dissoc.q2`; report §"task_oa_dissoc Q2".
  - [x] **Q3 two-field formalization: ORTHOGONAL fields, mechanism named** (2026-06-11).
    Over the same 7,760 scoped candidates: Spearman(flow-change, arrival-density) = −0.10
    (−0.09 citation-active-only) raw, +0.001/+0.009 after residualizing arrivals on the
    matching covariates — ZERO once density/dist-to-fg/growth are removed; top-50 Jaccard
    0.0, top-1% 0.013; 2x2 at top-decile thresholds (citation-active n=4,881): HH 36
    (vs ~49 under independence), HL 453, LH 477, LL 3,915. Quadrant faces (titles):
    HH = the neural-revolution front (Bahdanau/Luong attention-NMT, Collobert-Weston,
    encoder–decoder NMT — new canon AND new occupation); HL = the static canon /
    infrastructure (Zettlemoyer-Collins, SRL + joint-parsing classics; fills literally 0);
    LH = production zones (CLEF/QA, WMT submissions, applied summarization — arrivals
    without circuit change). Formal statement in the entry verdict above; one-liner added
    to curriculum-applications.md. Rows `task_oa_dissoc.q3`; report §"task_oa_dissoc Q3".

## Completed (Track 3)

- **Track 3: book gap-audit demo** (dispatched 2026-06-11 per
  `docs/handoffs/2026-06-11-track3-book-audit-brief.md`; **COMPLETED 2026-06-12,
  descriptive demo — below the validated frontier; 60 tests green** incl. 4 new contracts:
  LaTeX-strip rule, unit-split rule, fg/bg joint-embed identity, split-k full statistic ==
  GapScorer at k_fg==k_bg) — first non-wiki/OA fg, first production-shaped run of the
  validated suite. VERDICT: the suite produces an owner-judgeable missing-topics report —
  **12 coherent gap regions** (face-valid: random-matrix theory at d=0.62 from ch1 spectral
  = the strongest in-scope candidate; survey sampling; censoring/reliability; bandits/regret;
  noisy-label learning; TDA; multicollinearity estimators; …) + a **topic-grain headline**:
  the field's causal-inference (scope share 0.051), clustering (0.046), and time-series
  (0.045) literatures sit almost entirely beyond the book's topical reach (experimental
  design 0.116, bandits 0.124 next) — the curriculum-level absences. TWO method findings
  on record: (i) raw-r_F level-sets give ONE boundary-shell blob (size 1559, n_eff 1559,
  top-50 Jaccard 0.00 across configs) — pure-distance level sets are an annulus by
  construction; the validated FULL-statistic map fragments into 15 components at top-5%
  (7 at top-1%) with top-50 Jaccard 0.59–0.79 across k_fg; (ii) textbook-vs-abstracts GENRE
  confound: 141,317 of 156,111 fg-absent miner patterns are paper register ({proposed,
  simulation}…) — a df-share≥4%-in-AND-out-of-scope register rule leaves 14,794 vocab-gap
  patterns whose real labels are **approximate Bayesian inference**, **mixture models**,
  **nonparametric** (the t2 register caveat, now load-bearing at cross-genre fg/bg).
  Data cleaning on record: OA paratext types dropped 7,749 (dataset/peer-review records),
  dup-title 10,377, dup-TEXT 685 (publisher-blurb chapter piles whose r_B≈0 exploded hab
  to 1.4e12 — caught + fixed), 122,360 of 144,868 works used. Output
  `artifacts/gap_detection_eval/t3_book_audit.md` (ranked regions, each: top-terms label,
  5 exemplar titles, nearest book unit + distance, one-line owner framing); rows
  `results.json:task_t3`; report §"Track 3 — book gap-audit demo"; code `book_audit.py` +
  `run_t3_book_audit.py`; caches `cache/oa/t3_ds_book/` (725 pages x2 selects) +
  `cache/t3_commit.json`/`t3_topic_counts.json`. Phase log below.
  - [x] Phase A — fg extraction DONE (2026-06-12): **126 units** from 9 chapters (42
    \section + 93 \subsection + 9 intros, 17 tiny-unit merges; chars min/med/max
    211/519/4001). Unit rule: split at \section AND \subsection (42 bare \sections fall
    short of the brief's expected 80–150; subsection split lands in-range), chapter-intro
    units, <200-char units merge into the previous unit (coverage never dropped). LaTeX→text
    rule (8 steps: comments/wrapper/code/math/refs dropped, text commands unwrapped,
    captions survive) documented + tested in `book_audit.py`. 3 new toy-fixture tests
    (strip rule, unit split, joint-embed identity) — suite 59 green.
  - [x] Phase B — bg slice PRE-COMMITTED on counts (2026-06-12, recorded BEFORE any pull
    or scoring). 34 /topics searches → 221 distinct topics probed with the full works
    filter (`primary_topic.id,from/to_publication_date,has_abstract:true,language:en`;
    table `cache/t3_topic_counts.json`). Selection rule (fixed before probing): topic
    display_name names a METHODS area of statistics / statistical ML / DS foundations in a
    methods subfield (Stats&Prob, AI, Comp Theory, Signal Proc, MS&OR, Numerical) that a
    "Foundations of Data Science" curriculum covers or could plausibly cover; domain
    applications excluded (healthcare/imaging/text/security/materials/finance...); also
    excluded as applications-mass: T10320 Neural Networks and Applications 49K,
    T11689 Adversarial Robustness 29K, T11307 Domain Adaptation/Few-Shot 25K, T10500
    Sparse/Compressive Sensing 26K (engineering signal-recovery flavor). Window slid
    2015→**2019-01-01..2025-12-31** to land inside the brief's 50–150K (2015: 207K,
    2018: 162K, 2019: 144.9K) — no topic dropped to hit budget. **COMMITTED: 19 topics,
    combined meta.count = 144,868** (per-topic: T10136 StatMethods&Inference 15,423;
    T12535 ML&DataClassification 12,199; T12205 TimeSeries 11,654; T10968
    StatDistributionEstim 10,317; T10845 CausalInference 10,269; T11871 AdvStatMethods
    8,569; T11303 BayesModeling&Causal 8,025; T12101 Bandits 7,580; T11901 BayesMixture
    7,531; T12536 TopoGeomDataAnalysis 7,632; T11612 StochGradOpt 7,000; T12072
    ML&Algorithms 6,779; T12814 GP&BayesInference 6,267; T10637 Clustering 5,664; T10243
    StatMethods&BayesInf 5,191; T12056 MCMC 4,828; T11716 RandomMatrices 4,816; T11798
    OptimalExperimentalDesign 3,012; T13030 SurveySampling 2,112; record
    `cache/t3_commit.json`).
  - [x] Phase C — slice pulled (2026-06-12): 725 pages → **144,868 works** cached to
    `cache/oa/t3_ds_book/` (= the pre-committed meta.count exactly; polite pool, 418s,
    resumable cursor paging; reruns offline). Re-pulled once with `type` added to the
    select (cache key covers select fields; v1 shard kept) after paratext pollution
    surfaced in Phase D; integrity-verified (725 pages parse, cursor chain terminates,
    total exact).
  - [x] Phase D — embed + detectors DONE (2026-06-12): joint TF-IDF (vocab 19,241) →
    LSA-64 (EVR 0.102; LSA-128 sens EVR 0.148); scope q25 thr 1.018 → 30,590 of 122,360;
    k_fg=3/k_bg=10 split-k full statistic (identity-tested); level-set map top-5% → 15
    components (14 of size≥5); miner min_support 122 → 211,272 closed itemsets, register
    split as in the verdict; sensitivity (scope J / full-Spearman / top50 J): k=1
    0.605/0.725/0.587, k=5 0.79/0.945/0.786, LSA-128 0.684/0.883/0.471. Three banked
    iterations on the way (r_F-shell, register confound, dup-text hab explosion) — all on
    record in results.json + report.
  - [x] Phase E — outputs DONE (2026-06-12): `t3_book_audit.md` (12 framed regions +
    miner-label reading guide + topic-composition table + robustness), results.json:task_t3
    banked, report.md §TASK_T3_BOOK_AUDIT rendered, worklog closed. Suite 60 green.

## Curriculum Showdown (Phase 0)

- **Curriculum Showdown Phase 0: algorithmic-ToC generator + staged comparison** (dispatched
  2026-06-12 per `docs/handoffs/2026-06-12-curriculum-showdown-phase0.md`; **COMPLETED
  2026-06-12, 68 tests green** incl. 8 new toy-fixture contracts: k-rule
  determinism/feasibility/fallback, exact-DP ordering == brute force on the 3-cluster
  citation fixture, median-year tie-break, coverage/closure/violation metric forms, gap-bar
  + re-run-audit closure incl. a GapScorer-backed end-to-end pocket fixture).
  Goal: build `curriculum_toc.py` (algorithmic chapter/section ToC from corpus structure,
  banked t3 parts reused), run it on the cached t3 DS slice (TUNING topic — params frozen
  here), stage the 3-metric objective comparison vs the ds2 agent ToC, stage Phase-1 topics
  (counts only). NO LLM judging this phase. Rows → `results.json:task_showdown_p0`.
  **VERDICT (frozen generator config + the 3-metric suite, DS tuning topic):** generator =
  KMeans-**11** chapters (pre-committed feasible-max-silhouette rule, sil 0.07737; k≥12
  higher-sil configs all carry a <3% cluster, k=8 a >25% one) × 6–8 sections/chapter
  (same rule per chapter) over the t3 joint LSA-64; labels = cluster-wise t3 register
  split (+ on-record MathML-debris junk extension, labels only); order = exact min
  forward citation flow over **154,851** in-slice reference edges (bitmask DP,
  median-year tie-break). toc-algo vs ds2 agent ToC: field-mass coverage @ r=1.0181
  **0.993 vs 0.504**; gap-region closure (REVISED re-run-audit rule, on record — the
  pre-committed median-distance form was degenerate 12/12-both) **3/12 vs 0/12**
  (algo closes exactly the regions it gives whole chapters: lifetime distributions,
  bandits/regret, topology/TDA; ds2 = incumbent baseline 0 by construction);
  prereq-violation share **0.444 vs 0.497** (ds2 optimally re-ordered would still be
  0.476; cross-chapter edge bases 37,791 vs 97,620 — shares are the comparable number).
  The algorithmic ToC is face-valid foundational-first: optimization → bandits →
  spectral/probability theory → classical estimation/testing → TDA → supervised ML →
  lifetime distributions → Bayes/MCMC → causal inference → clustering → time series.
  Artifacts `showdown/toc_algo_datascience.md` + `showdown/side_by_side_ds2.md`; code
  `curriculum_toc.py` + `toc_metrics.py`; caches `cache/oa/ds_showdown_refs/` (725
  pages) + `cache/showdown_*`. Phase-1 staging: causal sub-slice verified standalone
  (8,390 works, rule-pick k=10); post-cutoff topic COMMITTED on counts = T10028 Topic
  Modeling (16,683 ≥ 2024-07-01, the only ≥15K candidate of 94 probed); executor
  cutoffs = gemma 3: 2024-08, gemma 4: 2025-01 (model cards). Phase log below.
  - [x] **PRE-COMMITTED RULES (recorded 2026-06-12 BEFORE any clustering/sweep output was
    looked at; the brief fixes the k-rule, the rest is instantiated here):**
    * **Corpus/embed**: the cached t3 slice, t3 cleaning verbatim (`assemble_bg` +
      nnz>=5 trim → expect 122,360), joint embed with the t3 fg (126 ds2 units) via
      `embed_joint(seed=42, LSA-64, t3 tfidf config)` — one shared space for clustering,
      both ToCs, and the 12 banked regions. fg rows are 0.1% of the TF-IDF universe
      (joint-fit caveat on record); clustering runs on bg rows ONLY.
      Recompute-validity gate: scope thr / n_scoped / 12 component sizes must reproduce
      banked task_t3 (1.0181 / 30,590 / [389,244,237,205,208,54,51,21,18,11,11,8]).
    * **Chapter k rule (brief)**: KMeans(n_init=10, random_state=42) on L2-normed LSA-64,
      k ∈ {8..14}; pick k = max mean silhouette (euclidean, sample_size=min(n,20000),
      random_state=42) subject to no cluster <3% or >25% of mass. INFEASIBLE-k fallback:
      fewest violating clusters, then higher silhouette. Sections: same rule within each
      chapter, k ∈ {3..8} (capped at n−1), mass shares relative to the chapter.
    * **Ordering rule**: refs absent from t3 cache (verified: selects carried no
      referenced_works) → re-pull same 19-topic filter, `select=id,publication_date,
      referenced_works` → `cache/oa/ds_showdown_refs/`. Flow F[a,b] = # citation edges
      (w in a) cites (v in b), both endpoints in the cleaned slice, a≠b. Chapter (and
      within-chapter section) order = exact min of forward mass Σ_{pos(a)<pos(b)} F[a,b]
      by bitmask DP (n≤16; deterministic tie-break: candidates scanned in ascending
      (median pub year, cluster id) order, first optimum kept). Refs unpullable ⇒
      median-year ascending + degradation flag.
    * **Labels**: miner-style top terms per cluster = `region_terms` log-odds over the
      2,000-keyword space, t3 register split ported cluster-wise (term REGISTER iff
      df-share ≥ 0.04 inside the cluster AND in the rest of the slice) + JUNK_TERMS
      dropped; 2 exemplar titles per section = nearest-to-centroid members (titles only).
    * **Metric suite (Phase-1-reusable, in `toc_metrics.py`, toy-tested); ToC
      representatives: algo = L2-normalized section centroids, ds2 = its 126 fg unit
      vectors (its real section texts, per brief):**
      1. *field-mass coverage at the t3 scope radius*: share of the 122,360 works with
         distance-to-NEAREST representative ≤ 1.0181 (banked t3 q25 thr; k=1 so neither
         ToC is tautologically at 0.25).
      2. *gap-region closure*: of the 12 recomputed task_t3 regions, count with MEDIAN
         member distance-to-nearest-representative ≤ 1.0181.
         **[REVISED 2026-06-12, on record]** the form above came back DEGENERATE when
         computed — 12/12 for BOTH ToCs — because the 12 regions live INSIDE the t3
         scope (bottom quartile of distance-to-fg) by construction; absolute k=1
         distance at the scope radius separates nothing (the t3 gaps are full-statistic
         peaks, not absolute-distance outliers). REVISED RULE, committed BEFORE its
         per-ToC numbers were computed: re-run the t3 audit statistic against the
         candidate ToC's representatives — full^T = hab·(Δ^T − 0.5σ^T) over the t3
         scoped pool with Δ^T from r_F(T's reps, k_fg=3), bg side (r_B, hab, k_bg=10)
         unchanged; gap bar = the t3 selection level (the minimum full-score inside the
         banked top-5% level set, i.e. the score at the boundary of gaphood); region
         CLOSED by T iff its members' mean full^T < bar — "the region would no longer
         qualify as a gap if the audit were re-run against this ToC". By construction
         the incumbent ds2 closes 0/12 (its region members were exactly the selected
         set) — the clean baseline. Degenerate first form stays banked for the record.
      3. *prereq violations*: assign every work to a chapter (algo = KMeans label;
         ds2 = chapter of nearest fg unit); violations = # cross-chapter citation edges
         (w cites v) with chapter(w) ordered BEFORE chapter(v); report count + share of
         cross-chapter edges. ds2 chapter order = ch1..ch9 as published.
    * **0b causal sub-slice**: T10845 rows of the cached slice, t3 cleaning; standalone
      embed (t3 tfidf config, LSA-64, seed 42); chapter sweep k ∈ {5..10} (the brief's
      <15K-docs rescale); record survivor count + sweep table only, NO ToC, NO labels.
    * **0b post-cutoff topic**: probe /topics for {LLM agents, retrieval-augmented
      generation, mechanistic interpretability, diffusion models}; filter
      `primary_topic.id:X,from_publication_date:2024-07-01,has_abstract:true,language:en`;
      meta.count first; pick ≥15K-works candidate, committed on counts + one-line
      structural-novelty rationale BEFORE any pull (pull is Phase 1).
  - [x] **Executor cutoff finding** (2026-06-12, official model cards, web read-only — the
    GPU/local model was NOT queried): Gemma 3 model card
    (ai.google.dev/gemma/docs/core/model_card_3): "The knowledge cutoff date for the
    training data was **August 2024**." Gemma 4 model card (model_card_4, released
    2026-04): pre-training data "cutoff date of **January 2025**". ⇒ "post-cutoff" for
    our executor class = strictly after 2024-08 (gemma 3) / after 2025-01 (gemma 4). The
    0b topic window `from_publication_date:2024-07-01` overlaps gemma 3's tail by ~2
    months and gemma 4's by ~6 — the bulk of any ≥15K slice is later, but Phase 1 should
    prefer the 2025+ sub-mass when it wants a strictly-post-cutoff read.
  - [x] Phase S-A′ — embed + anchor gate DONE (2026-06-12): cleaning 144,868 → 122,519
    eligible → 122,360 used (exact banked match); LSA-64 EVR 0.102; thr 1.0181, scoped
    30,590; 12 recomputed region sizes == banked task_t3 exactly
    ([389,244,237,205,208,54,51,21,18,11,11,8]). Caches under
    `artifacts/gap_detection_eval/cache/showdown_*`. Banked `.embed_anchor`.
  - [x] Phase S-A — refs pull DONE (2026-06-12): 725 pages → **144,868 works** with
    `id,publication_date,referenced_works` to `cache/oa/ds_showdown_refs/` (= the t3
    meta.count EXACTLY — zero drift since the t3 pull; polite pool, 226s, resumable).
  - [x] **0b post-cutoff topic COMMITTED on counts** (2026-06-12, BEFORE any pull; pull is
    Phase 1): **T10028 "Topic Modeling" (AI), 16,683 works** at
    `from_publication_date:2024-07-01,has_abstract:true,language:en` — the ONLY candidate
    ≥15K across the brief's four searches + 16 taxonomy-name + 6 due-diligence queries
    (37 distinct ≥10K-total topics probed; runner-up T10181 NLP Techniques 14,043 < 15K;
    XAI home of mech-interp 6,647; GAN/diffusion image-synthesis home 5,157; the literal
    "LLM agents"/"RAG"/"mech interp" strings have NO /topics hit — the fixed taxonomy
    predates those fields). RATIONALE (one line): T10028 is where OpenAlex files the
    LLM/RAG/agents literature, so its post-2024-07 mass is dominated by a structure
    (instruction-tuned LLM applications, retrieval pipelines, agent papers) that did not
    exist pre-2023 and post-dates the gemma-3 cutoff (2024-08) almost entirely.
    Banked `.postcutoff_candidates` (rows incl. due-diligence probes).
  - [x] Phase S-B — chapter k sweep DONE (2026-06-12; scores recorded BEFORE any labels
    were computed — no label step has run yet). Silhouette (20K seeded subsample) / share
    min–max: k=8 0.06653 / 0.0483–**0.2602 infeasible**; k=9 0.07125 / 0.0478–0.1968;
    k=10 0.07435 / 0.0476–0.1957; k=11 **0.07737** / 0.0370–0.1936; k=12 0.08147 /
    **0.0162 infeasible**–0.1856; k=13 0.08609 / 0.0162 inf.; k=14 0.08858 / 0.0161 inf.
    **RULE PICK: k=11** (feasible-max-silhouette; the higher-sil k≥12 all carry a <3%
    cluster, k=8 carries a >25% cluster). Banked `.chapter_sweep` +
    `cache/showdown_sweep.json`.
  - [x] Phase S-C — ToC built (2026-06-12): 11 chapters / 78 sections
    (per-chapter section k by the same rule: 6,6,8,8,7,8,8,6,8,6,7 in curriculum order);
    citation ordering: optimal forward share 0.444 (arbitrary-id order 0.526); chapter
    cluster-id order [7,6,5,10,2,0,1,9,3,8,4]; deterministic re-run reproduced order +
    flow exactly. One on-record labeling-hygiene iteration: MathML/XML namespace debris
    ({mathml, xmlns, mml, xlink, …}) surfaced as labels of two ch-3 sections → added to
    an EXTRA_JUNK set (same class as t3's LaTeX-debris JUNK_TERMS); LABELS ONLY — k,
    clusters, ordering untouched. Exemplar near-dup titles also deduped (presentation).
    `showdown/toc_algo_datascience.md` written with provenance footer (params hash,
    corpus, seed, date, NO-LLM note). Banked `.toc_build`.
  - [x] Phase S-D — metrics DONE (2026-06-12): coverage 0.993 vs 0.504; closure
    (revised rule, see the [REVISED] block above) 3/12 vs 0/12 — algo closes t3 regions
    ranked 1 (lifetime distributions → its Ch7), 7 (bandits/regret → Ch2), 8 (algebraic
    topology → Ch5); per-region mean full-scores banked. Prereq violations 16,784/37,791
    (0.444) vs 48,473/97,620 (0.497); ds2's own partition optimally re-ordered = 0.476.
    `showdown/side_by_side_ds2.md` written (table + both ToCs verbatim + caveat notes:
    coverage is construction-favored for corpus-derived ToCs; closure baseline is 0 for
    the incumbent; shares not raw counts are comparable for violations). Banked `.metrics`.
  - [x] **0b causal (niche) topic VERIFIED standalone** (2026-06-12): T10845 Advanced
    Causal Inference = **8,390** cleaned works inside the slice-cleaned corpus (10,269
    raw; t3 cleaning), standalone t3-config embed vocab 5,388 / LSA-64 EVR 0.133 /
    nnz≥5 keeps 8,378; cluster-ability sweep k∈{5..10} (the pre-committed <15K rescale):
    sil 0.0374→0.0494 monotone, k=5–6 infeasible (max share 0.326/0.292), k=7–10
    feasible, **rule would pick k=10 (sil 0.04943)** — it stands alone. NO ToC generated
    (Phase 1, frozen params). Banked `.causal_subslice`.
  - [x] Phase S-E — 0b staging DONE (2026-06-12): the three [x] entries above (causal
    sub-slice verified; post-cutoff pick committed on counts, 94 candidate rows banked;
    executor cutoff findings from official model cards). Banked `.causal_subslice` +
    `.postcutoff_candidates`. Suite 68 green at close.

## Curriculum Showdown (Phase 1)

- **Curriculum Showdown Phase 1: blind 2×2 ToC tournament** (dispatched 2026-06-12 per
  `docs/handoffs/2026-06-12-curriculum-showdown-phase1.md`). 3 topics × 3 arms
  (algo = frozen Phase-0 generator; agent = gemma4:31b-it-qat via pi, web on, best-of-3
  internal pick; hybrid = gemma polish of the algo skeleton), provenance-blind pairwise
  judging + the objective metric suite. Rows → `results.json:task_showdown_p1`; code
  `artifacts_scripts/gap_detection/showdown_p1.py` (book_audit/curriculum_toc/toc_metrics
  reused READ-ONLY; `fit_embed` replicates `embed_joint` verbatim + returns the fitted
  vectorizer/SVD — equivalence contract-tested + anchor-asserted).
  - [x] **POST-CUTOFF RE-CUT LOCKED on counts (2026-06-12, BEFORE any pull)**: window
    `from_publication_date:2025-02-01` (+has_abstract:true, language:en; gemma-4 cutoff
    Jan 2025 per Phase 0). Counts: **T10028 Topic Modeling 9,944** ≥ 8K → **LOCKED:
    T10028, window 2025-02-01→open**; fallbacks not needed (T10181 NLP 6,909; union
    16,853 — recorded). Banked `.postcutoff_recut`. Pull select = the brief's six fields
    + `type` (the ONE deviation, required to apply the t3 KEEP_TYPES cleaning verbatim)
    → `cache/oa/showdown_pc_2025/`.
  - [x] **PRE-COMMITTED Phase-1 protocol (recorded BEFORE the pull, BEFORE any agent/
    hybrid run, BEFORE any judgment or metric computation):**
    * **Topic statements + audiences (same text to EVERY arm and the judge); H1 of all
      9 candidates = "Table of Contents: <Topic Title>"**: datascience = "Data Science",
      audience "quantitative-major undergraduates (calculus, linear algebra,
      programming)"; causal = "Causal Inference", audience "graduate students /
      quantitative researchers entering causal inference from a general stats
      background"; postcutoff = "Topic Modeling" framed as "the field as it stands in
      2026 — the literature published since early 2025", audience
      "researchers/practitioners entering this literature in 2026".
    * **Algo arm**: frozen Phase-0 rules. DS = the Phase-0 ToC re-emitted verbatim
      (neutral H1, provenance stripped). Causal + postcutoff: standalone t3-config
      embeds (seed 42, LSA-64), chapter k∈{5..10} when the cleaned slice <15K docs
      (else {8..14}), sections k∈{3..8}, same silhouette+share rule, register-filtered
      term labels Title-Cased but NOT hand-polished, exact min-forward-flow ordering
      (median-year tie-break) over the topic's own in-slice citation edges (causal:
      ds_showdown_refs sub-slice edges; postcutoff: its own referenced_works).
    * **Agent arm**: 3 pi_launch mini-runs per topic (angles problems-first /
      methods-first / applications-first), gemma4:31b-it-qat, thinking high, web via
      staged websearch.py/fetch_url.py, auto_continue.max=3 until `test -s toc.md`;
      then ONE non-blind internal judge run per topic picks the best of the 3 (the
      agent arm's own tournament). Park rule: a mini-run unusable after 3 nudges
      contributes its best output as-is (noted).
    * **Hybrid arm**: one pi mini-run per topic; input = the algo skeleton
      (chapter/section term-labels + 2 exemplar titles each + the fixed order); rules
      stated verbatim in its brief: rename chapters/sections into real titles, write
      the intents, MAY merge at most 2 adjacent chapters, MUST NOT reorder or move
      content between chapters.
    * **Blind judging protocol**: per topic all 3 pairs (algo,agent), (algo,hybrid),
      (agent,hybrid); each pair judged 4× = both presentation orders × 2 samples.
      pi exposes NO sampler-seed control → "2 sampling seeds" is implemented as 2
      independent samples at provider-default sampling, labeled s1/s2 (recorded
      limitation). A/B assignment: numpy default_rng(42); topics in order
      [datascience, causal, postcutoff] × pairs in the order above; draw u=integers(2):
      u=0 → the pair's first-listed arm is A in order-1, u=1 → the second; order-2 =
      swapped; judgments = (order1,s1),(order1,s2),(order2,s1),(order2,s2). Judge =
      gemma4:31b-it-qat via pi, offline, NO web tools, thinking high; candidates
      provenance-stripped, labeled A/B, identical H1s. Per-pair score = majority of
      the 4; 2–2 = TIE (reported as such, 0.5–0.5 in any aggregate); full 4-way table
      reported. **Rubric (given to the judge verbatim):** "Judge the two candidate
      tables of contents on exactly four criteria, weighted equally: (1) Coverage for
      the stated audience: does the ToC span what this audience needs for this topic?
      (2) Progression: prerequisites precede their use — no chapter depends on
      material that appears later. (3) Pedagogical arc: the sequence forms a coherent
      learning trajectory from foundations to frontier. (4) Feasibility at depth:
      could a chapter drafter write substantive, non-filler chapters from these
      intents? Then declare a winner. Write ./verdict.md: first line exactly
      'WINNER: A' or 'WINNER: B'; then exactly 3 rationale lines (one sentence each)."
    * **Objective metrics (every candidate; toc_metrics.py reused)**: ToC reps: algo =
      its L2-normed section centroids (Phase-0 convention); agent/hybrid = the
      unit-normalized frozen-space images of each section's "TITLE. INTENT" text via
      the topic's FITTED vectorizer+SVD (the ds2 mapping analogue); sections whose
      intent maps to a zero TF-IDF image are DROPPED from the rep set (count
      reported) — a zero vector would sit ~distance 1 from every unit doc and fake
      coverage. (1) field-mass coverage, k=1: DS radius = banked 1.0181; causal/
      postcutoff have NO banked fg-based threshold → radius = the SAME numeric 1.0181
      pre-committed as a fixed angular yardstick in unit-norm LSA-64 space; coverage
      compared WITHIN a topic only, never across topics; secondary radius-free
      statistic also reported: median nearest-rep distance per arm (guards against
      saturation degeneracy). (2) prereq-violation share: every doc assigned to a
      chapter (algo = its KMeans label; agent/hybrid = chapter of the nearest kept
      section rep); violations = forward citation mass under the ToC's published
      chapter order (algo = its built order) / cross-chapter mass; context number =
      the DP-optimal share for that partition (k≤16 only). (3) gap-region closure:
      DATA-SCIENCE ONLY, the 12 banked task_t3 regions, REVISED re-run-audit rule
      (full^T vs the candidate's reps, bar = banked t3 selection level). Caveats
      carried from Phase 0: coverage structurally favors the algo arm (its sections
      tile the corpus); the closure incumbent caveat applied to ds2, which is NOT an
      arm here — fresh candidates are construction-unbiased; interpret metrics WITH
      the judge lens, never alone.
  - [x] Phase P1-A — corpora DONE (2026-06-12): post-cutoff pull 50 pages → 9,944 works
    (meta.count = the locked 9,944, zero drift) → t3 cleaning → **9,176 eligible
    (paratext 37, short 64, non-Eng 22, dup-title 634, dup-text 11) → 9,157 used**
    (nnz≥5), vocab 6,103, LSA-64 EVR 0.127; in-slice citation edges **697** of 112,916
    reference entries — the post-cutoff regime structurally lacks in-window citation
    signal (16-month window + publication lag), so its ordering leans on the
    pre-committed median-year/cluster-id tie-break (recorded limitation). Causal:
    8,390 cleaned (= banked), standalone vocab 5,388 / EVR 0.133 / keeps 8,378
    (= banked), in-sub-slice edges **12,681**. DS: joint embed REFIT reproduces the
    Phase-0 cached arrays EXACTLY (max |ΔZ| = 0.0, 122,360 bg rows); fitted
    tfidf+svd cached per topic (`cache/p1_*_vectorizer.joblib`) for intent mapping.
    Banked `.corpus_*`, `.postcutoff_pull`.
  - [x] Phase P1-B(algo) — algo ToCs DONE (2026-06-12, frozen rules): causal k∈{5..10}
    → **k=10** (feasible-max-sil 0.04943 — same pick the Phase-0 standalone sweep
    predicted), 76 sections, exact-DP order forward share **0.398** (arbitrary 0.515),
    12,681 edges; postcutoff k∈{5..10} → **k=10** (feasible-max-sil), 69 sections,
    forward share **0.277** (arbitrary 0.382) on the thin 697-edge flow. Chapter
    content of the post-cutoff algo ToC is visibly the 2025+ LLM literature (CoT/RL,
    quantization/throughput, LoRA/DPO, RAG/retrievers, NER, summarization, clinical
    NLP, embeddings, ethics, claims/vulnerabilities). `p1_{causal,postcutoff}_algo.md`
    + `p1_datascience_algo.md` (Phase-0 ToC re-emitted, neutral H1) + 3 hybrid
    skeletons written. Banked `.algo_tocs`.
  - [x] Phase P1-B(agent+hybrid) mini-runs DONE (2026-06-12 06:25): 12 pi_launch runs
    (gemma4:31b-it-qat, thinking high, web tools staged), all 12 produced parseable
    toc.md on the first pass (0 auto-continues): agent minis 6–14 chapters / 24–68
    sections; hybrids EXACTLY mirror the algo skeleton shapes (ds 11ch/78s, causal
    10ch/76s, pc 10ch/69s — the may-merge option was not exercised; no reorders).
    Web usage confirmed (13 bash tool calls/run; websearch+fetch hits in events).
    ONE incident: mini_causal_applications got an external SIGTERM ~3.5 min in
    (rc=143, stderr clean — environment reaper, not a model failure); clean retry
    succeeded. Run dirs (under ~/src/learn-and-teach/outputs/2026-06-12/): ds minis
    05-32-44 / 05-35-34 / 05-39-52; causal minis 05-42-22 / 05-46-20 / 06-21-21
    (retry; first attempt 05-48-54 killed); pc minis 05-52-24 / 05-59-16 / 06-02-46;
    hybrids 06-05-38 / 06-11-51 / 06-15-53. Manifest `showdown/p1_runs.tsv`; briefs
    under `~/src/learn-and-teach/outputs/showdown_p1/briefs/`. Internal picks: below.
  - [x] Phase P1-B(picks) — internal agent-arm picks DONE (2026-06-12 ~06:45, one pi
    run per topic, offline, NOT blind by design): datascience → draft 2
    (methods-first, 14ch/68s); causal → draft 1 (problems-first, 8ch/37s);
    postcutoff → draft 2 (methods-first, 8ch/32s). p1_<t>_agent.md +
    p1_<t>_hybrid.md written (H1 normalized to the locked neutral form); banked
    `.collect`. Pick run dirs in `showdown/p1_picks.tsv`. Tooling note (on record):
    p1_run_jobs.sh needed `</dev/null` on the pi invocation (a `while read` loop
    feeds stdin to children — first run consumed the jobs file) + absolute-path
    canonicalization; both fixed before any judgment ran.
  - [x] Phase P1-C — blind judging DONE (2026-06-12 ~07:00): 36/36 judgments
    completed, 0 invalid verdicts, 0 failures (~45s each, offline, no web tools).
    **Win matrix is a uniform sweep: agent > hybrid > algo in every topic, every
    pair 4–0** (datascience, causal, postcutoff identical). Validity notes: (1)
    verdicts are STABLE across both presentation orders and both samples (no
    position bias — a judge that always picked 'A' would have split 2–2); (2)
    blinding is porous BY CONSTRUCTION: the judge repeatedly identifies the algo
    arm's register ("automated literature clustering report", "systematic
    bibliography") — inherent to the locked no-hand-polish rule, on record; (3)
    the postcutoff rationales judge TOPIC SCOPE from the model's prior of "Topic
    Modeling" (rewarding classic-LDA-to-2026-frontier framing, penalizing the
    corpus-shaped candidates for the RAG/agents/compression content the post-2025
    slice ACTUALLY contains) — the cleanest observed instance of judge
    prior-bias. Banked `.judging`; verdict run dirs in `showdown/p1_judges.tsv`.
  - [x] Phase P1-D — metrics + results DONE (2026-06-12): all 9 candidates, 0
    zero-dropped sections, parser fallbacks 0. Coverage@1.0181 (algo|hybrid|agent):
    ds 0.993|0.671|0.496; causal 0.997|0.654|0.310; postcutoff 0.996|0.611|0.117.
    Median nearest-rep distance same ordering everywhere. Prereq-violation share:
    ds 0.444|0.460|0.474; causal 0.398|0.466|0.458; postcutoff 0.277|0.511|0.584.
    DS gap-closure 3/12|1/12|0/12 (fresh candidates construction-unbiased; the
    agent closes none, the hybrid closes the lifetime-distributions region).
    **Disagreement table: 20 of 21 comparable judge-vs-metric cells disagree**
    (9 pairs × 3 metrics = 27, minus 6 closure-N/A on non-DS topics; the single
    agreeing cell = causal agent-v-hybrid violations, where agent leads both) —
    judge picks the LLM arm everywhere; every other metric leader is algo (or
    hybrid over agent).
    **VERDICT: at the judge lens NO interaction (LLM sweep); at the metric lens
    the predicted interaction appears as a monotone gradient — agent coverage of
    actual field mass decays canonical→niche→post-cutoff 0.50→0.31→0.12 (same
    construction throughout, so the GRADIENT is internally fair) while algo holds
    ~0.99 and hybrid recovers about half the deficit; agent ordering degrades
    0.47→0.58 violation share while algo improves to 0.28. The hybrid is the only
    arm that wins under both lenses somewhere (judge-preferred over algo
    everywhere, metric-stronger than agent everywhere).** Artifacts:
    `showdown/p1_results.md` (win matrix + 4-way tables + metric table +
    disagreement table + rationale excerpts + plain-language verdict), 9
    candidates `showdown/p1_<topic>_{algo,agent,hybrid}.md`, banked
    `.metrics`/`.judging`/`.results_summary`/`.collect`/`.judge_assignments`.
    **Suite 74 green** (68 + 6 new toy contracts: fit_embed≡embed_joint,
    transform-roundtrip + zero-image guard, toc.md parser fixture, seeded A/B
    blinding determinism+swap, majority/tie + winner extraction,
    nearest-rep chapter assignment). Phase 1 CLOSED.

## Curriculum Showdown (R0 round-trip)

- **Curriculum Showdown R0: round-trip operators E/D, M-side cells** (dispatched 2026-06-12 per
  `docs/handoffs/2026-06-12-showdown-r0-roundtrip.md`). **COMPLETED 2026-06-12** — compute banked
  to results.json; report `showdown/r0_roundtrip.md` authored by the dispatcher (the agent parked
  on a monitor after the score phase and did not wake to write it). **Headline:** (1) alignment
  energy δ(T,R_MM-free(T)) for the agent ToCs = **0.604 / 0.744 / 0.795** (ds/causal/postcutoff) —
  reproduces the P1 ranking with NO judge and NO reference; algo ≤0.27, hybrid between; the
  reusable intrinsic score. (2) **R_AM improvement grows as the prior thins** (ordering axis, the
  unconfounded one): agent prereq-violation share ds 0.474→0.464 (≈ its 0.441 floor — mechanics
  add nothing) vs postcutoff 0.584→**0.397** respect / 0.290 free. (3) Idempotence ~0.95 respect;
  **closure 0.9475 ≈ c-TF-IDF 0.9454** (c-TF-IDF wins free 0.894 vs 0.848) → use c-TF-IDF, the FCA
  closure earns nothing here. (4) **TWO coverage degeneracies**: free=union-extent AND
  respect=whole-corpus partition both inflate coverage to ~0.99 → coverage is unusable as an R_AM
  metric; ordering + energy are the trustworthy axes. Iceberg fix = task #29. Build E_M (closure +
  c-TF-IDF extractors) and D_M (respect-partition + free modes) over the FROZEN Phase-0/1
  spaces and caches; compute R_MM idempotence (3 algo ToCs), R_AM improvement (3 agent ToCs,
  both modes, banked metric suite, derived ToCs emitted), round-trip fidelity tables
  (2 extractors × 2 modes × 9 candidates), alignment energy for all 9, extractor arbitration.
  Code `artifacts_scripts/gap_detection/roundtrip.py`; rows → `results.json:task_showdown_r0`;
  CPU only, no network, everything from `cache/p1_*` + `cache/showdown_*`.
  - [x] **PRE-COMMITTED CONSTRUCTIONS (recorded 2026-06-12 BEFORE any extraction/derivation/
    cell was computed; the brief fixes minsup≈25% and cap≈40 — instantiated exactly here):**
    * **Doc-assignment A(T)** (the P1 metric convention reused verbatim, `_candidate_reps`):
      algo arm = its native KMeans partition (chapter = cluster at its published position);
      agent/hybrid = every corpus doc → chapter of its nearest kept section rep (frozen-space
      images of 'TITLE. INTENT' via the topic's cached vectorizer+SVD; zero-image sections
      dropped). Spaces: DS joint LSA-64 (122,360 bg rows), causal/postcutoff standalone LSA-64
      (8,378 / 9,157 rows). Chapters indexed by PUBLISHED position 0..k−1 throughout.
    * **E_M, closure variant** (the miner machinery, register-filtered, iceberg): for chapter
      doc set D over the topic's binary keyword matrix K (top-2000-df vocab): (1) drop
      JUNK_TERMS ∪ EXTRA_JUNK columns; drop REGISTER columns for D = terms with df-share ≥
      0.04 BOTH inside D and in corpus∖D (the t3 register rule ported chapter-wise);
      (2) minsup = max(2, ceil(0.25·|D|)); (3) family = closed_filter(mine_frequent_itemsets(
      K[D, kept], min_support=minsup, max_len=2)) — detectors.py verbatim; (4) keyword set =
      the family's TERM UNION ranked by within-D support desc (tie: vocab index asc), capped
      at 40. (The term union of the closed family equals the register-filtered iceberg
      1-itemsets — the practical FCA intent at minsup; closed-family size banked as a
      diagnostic.) |D| < 2 or nothing surviving ⇒ empty set (counted, scored 0).
    * **E_M, c-TF-IDF variant** (the P1b comparator, same filters): same junk+register column
      drop; classes = the candidate's chapter doc sets; tf(t,c) = within-chapter df count
      (binary K); W(t,c) = tf(t,c)/Σ_t tf(t,c) · ln(1 + A/f_t), f_t = full-topic-corpus df of
      t, A = mean over the candidate's chapters of Σ_t tf(t,c); top-40 by W desc (tie: vocab
      index asc). Register stats and f_t are always relative to the FULL topic corpus, in both
      modes — E_M is a function of (D, corpus) only.
    * **Extent rules (D_M)**: match(d, K_i) = |terms(d) ∩ K_i| / |K_i|. RESPECT-PARTITION:
      d → argmax_i match(d, K_i) among chapters with match > 0; ties → the chapter EARLIEST in
      the candidate's published order; docs with zero match to every K_i = UNASSIGNED
      (excluded from every extent; counted). Chapters with empty extent are DROPPED from the
      derived ToC (counted; their respect-fidelity contribution = 0). FREE: extent(pool) =
      {d : |terms(d) ∩ ∪_i K_i| ≥ 1} — the induced sub-corpus.
    * **D_M derivation (frozen generator rules only)**: RESPECT keeps chapter identities =
      extent(K_i); re-derives sections (KMeans k ∈ {3..min(8, n−1)} by the frozen
      silhouette+share rule; n < 8 ⇒ one section = the chapter), labels (register-filtered
      `cluster_terms`), exemplars (nearest-to-centroid, title-deduped), and ORDERING — exact
      min-forward-flow bitmask DP over the extent partition's cross-chapter citation flow,
      median-year tie-break; within-chapter section DP likewise. FREE: the frozen Phase-1
      generator (`build_toc`) end-to-end on the induced sub-corpus — chapter k ∈ {8..14} if
      n_extent ≥ 15K else {5..10}, sections {3..8}, same labels/ordering; sub-corpus citation
      edges remapped from the topic's cached edge list. Everything seed 42.
    * **Round-trip fidelity**: re-extract with the SAME extractor from the derived ToC's
      chapter doc sets (respect: the extent partition; free: the new KMeans partition).
      Respect fidelity = mean_i J(K_i, E(extent(K_i))), identity-matched. Free fidelity =
      mean_i max_j J(K_i, K'_j) (per-original-chapter best match). J(∅,∅) := 1, J(A,∅) := 0.
      Jaccard is over term SETS (cap-40 members).
    * **Alignment energy (one formula, on record)**: δ(T) = 1 − mean_i max_j J(K_i(T),
      K_j(R_MM-free(T))) under the CLOSURE extractor (headline; the c-TF-IDF column reported
      as sensitivity), PAIRED with Δcoverage = coverage@1.0181(R_MM-free(T)) −
      coverage@1.0181(T) over the full topic corpus. **Hypothesis on record:** agent-ToC δ
      rises canonical → niche → post-cutoff (intrinsic, judge-free reproduction of the P1
      gradient).
    * **Metric suite on derived ToCs (R_AM cells)**: banked P1 forms verbatim — reps =
      L2-normed section centroids; coverage@1.0181 (k=1) over the full topic corpus; median
      nearest-rep distance; prereq-violation share with doc→chapter = nearest-section-rep
      chapter (uniform derived-ToC convention = the P1 agent convention) under the DERIVED
      chapter order; DS only: gap-region closure, re-run-audit rule vs the banked t3 bar.
      Each derived agent ToC scored NEXT TO its original (original rows recomputed with the
      same code path). Derived ToCs emitted from the CLOSURE-fed derivations only (the
      brief's filename template) → `showdown/r0_<topic>_agent_RAM_{respect,free}.md`,
      toc.md shape, neutral H1 (P1 writer reused); c-TF-IDF-fed derivations are computed and
      scored for arbitration cell (b) but not emitted.
    * **Extractor arbitration rule**: (a) M-side idempotence = R_MM respect+free fidelity on
      the 3 algo ToCs; (b) stability of R_AM deltas = per-metric (coverage, violation share)
      delta (derived − original) agreement between closure-fed and ctfidf-fed derivations
      (mean |Δ_closure − Δ_ctfidf| + sign agreement, 3 agent topics × 2 modes). Recommendation
      recorded for downstream phases; P1b's own kw cells stay c-TF-IDF (locked upstream).
  - [x] **[REVISED 2026-06-12, on record — minsup share 0.25 → 0.05]** The pre-committed
    closure E_M came back DEGENERATE at the extraction step (BEFORE any round-trip cell was
    computed): set sizes 0–6, many chapters EMPTY (banked
    `.extraction_minsup25_degenerate`). Diagnosis: the 25% iceberg and the 0.04/0.04
    register rule interact — chapter-DISTINCTIVE terms sit at 5–20% within-chapter df
    (below 25%), while terms above 25% are mostly genre register (≥4% outside too) and are
    correctly dropped; the suggestion-level 25% knob was tuned for no real corpus.
    Revision rule (stated before re-running, probe = SET SIZES ONLY, no downstream cell
    touched): minsup share = the LARGEST s ∈ {0.25, 0.20, 0.15, 0.10, 0.05} for which EVERY
    candidate chapter retains ≥5 surviving terms pre-cap. Probe (min/med/max surviving
    terms across chapters): s=0.10 fails (postcutoff/agent min 0; ds/agent min 2);
    **s=0.05 passes everywhere** (mins 25–48, medians 28–77, maxes 52–136 — the 40-cap now
    binds as the brief intended). REVISED: minsup = max(2, ceil(0.05·|D|)), all else
    unchanged (register rule, closed-itemset machinery, support-desc cap-40). The toy
    fixtures are share-invariant (floor 2 dominates at n=8) — contracts unchanged.
  - [x] Code + toy contracts DONE (2026-06-12): `roundtrip.py` (phases extract / derive /
    score / report; per-derivation pickle cache under `cache/roundtrip/`) + 6 new
    toy-fixture tests (Galois extent/intent on a clean block context incl. the
    minsup-boundary straddler term; chapter-level closure idempotence E∘D∘E = E for BOTH
    extractors; extent assignment determinism incl. tie→earliest-published and zero-match
    →unassigned; respect-vs-free mode contract — respect keeps identities and reorders by
    citation flow [published B,A → derived A,B], free re-picks k by the frozen rule on a
    5-blob fixture; c-TF-IDF top-terms sanity; fidelity/energy form edge cases).
    **Suite 80 green** (74 at brief time + 6).
  - [x] Phase R0-A — extraction DONE (2026-06-12, revised minsup): closure set sizes now
    25–40 across all 9 candidates (cap binds; e.g. ds/agent k=14 sizes 26–40, causal/agent
    k=8 sizes 27–40); c-TF-IDF exactly 40 everywhere. Doc-assignment = the P1
    `_candidate_reps` convention. Banked `.extraction` (+ `.extraction_minsup25_degenerate`
    kept for the record).
  - [x] Phase R0-B — all 36 derivations DONE (2026-06-12, 1,277s worker, CPU ≤8 threads,
    per-item pickle cache `cache/roundtrip/d_*.pkl`): 9 candidates × 2 extractors × 2
    modes, zero failures. Shape notes: DS free extents ≈ 122.0K of 122,360 (the ≥1-hit
    pool extent sees ~99.7% of the corpus; closure vs ctfidf pools differ by ~10 docs and
    give near-identical KMeans structure); DS respect (algo/closure) leaves 326 docs
    unassigned and drops 0 chapters; causal agent free extent 8,290/8,378 with the
    generator re-picking k=10 (agent published 8); ds agent free re-picks k=11 (agent
    published 14).

## In progress

*Net design state after P1–P4 + both temporal holdouts + the Track-3 demo (60 tests green):
Track 1 = species-matched suite (fused r_F / miner / voltage), no detector subsumes another,
fusion is real. Track 2 validated end-to-end on TWO real corpora (wiki + OA NLP) — absolute
level-set components are the arrival-prediction signal in both; P2 residual components + P3a
change field are validated change-localizers. Track 3 demo: the single-snapshot descriptive
suite ships an owner-judgeable book audit; cross-genre fg/bg needs the register rule, and
discovery maps need the full statistic, not raw r_F.*

*(Track 1 finish moved to Completed 2026-06-11; Track 3 demo completed 2026-06-12.)*

## Curriculum Showdown (Phase 1b — cluster web-off ToC generation)

- **Owner enabled the Stony Brook AI SLURM cluster (host `ai`, quadro 48 GB GPUs) to run gemma
  showdown work in PARALLEL with the local box** (2026-06-12). Infra banked in memory
  `ai-cluster-gemma.md`: rootless ollama + gemma4:31b-it-qat staged to shared NFS; each job runs
  `ollama serve` warm on a quadro node. Courtesy cap (owner): quadro1/quadro2 only, ≤2 jobs/node,
  ≤4 total.
- **Web-off ToC generation DONE — 18 ToCs** (`artifacts/gap_detection_eval/showdown/p1b_*_weboff.md`):
  ds/causal/postcutoff each = agent-NL ×3 angles + kw + hybrid (5); bandits (detector-selected 4th
  topic) = agent-NL ×3. Raw ollama is CLOSED-BOOK, so these ARE the NL-web-off / kw-web-off cells;
  web-ON cells still need local `pi`+websearch or a fetch step. Generation only — scoring pending.
- **kw-cell input PRE-COMMIT** (recorded before any scoring): keyword-prompt list = top-40 terms of
  the topic's FROZEN vocab with corpus df-share ∈ [0.02, 0.40], minus JUNK ∪ a fixed academic-register
  stoplist (propose/approach/method/paper/results/show/… + generic-ML data/model/learning/algorithm).
  Distinct from R0's per-chapter register mask (which needs an inside/outside contrast a corpus-level
  list lacks). Script: prep_inputs.py.
- Hybrid cells first collapsed to 3–4 ch (a 6000-char skeleton-truncation bug in prep); fixed with the
  full ~22 KB skeleton + "keep all chapters" → 11/10/10 ch, matching the algo skeletons.
- REMAINING for full Phase 1b: (a) **bandits completion** — OpenAlex pull + frozen-generator algo ToC
  + kw list (needs the bookgap pipeline on the cluster; cluster has internet) → then bandits kw/hybrid
  cells; (b) **web-ON cells** (kw-web-on); (c) **metrics** (toc_metrics coverage/violation +
  term-anchoring) + **blind judging** over all cells — CPU, local.
- **Web-off metrics DONE** (`showdown/p1b_weboff_metrics.md`, n=1/cell): (1) kw-prompting raises corpus
  keyword-recall hugely (0.47→0.88 mean) but coverage only on POST-CUTOFF (kw 0.357 > NL 0.209) — kw is
  ≈NL on canonical, better past the frontier, always more anchored → "corpus-in book-out" viable with an
  interaction. (2) **Hybrid dominates coverage** (0.57–0.68, ~2× NL/kw, > P1 web-on NL) — reaffirms
  hybrid as the production answer, BUT partly mechanical (inherits the algo skeleton) → needs judging.
  (3) Web helps canonical (ds NL-off 0.33 < on 0.50), not niche (post-cutoff off 0.21 > on 0.12).
- **Blind judging DONE** (`showdown/p1b_weboff_readout.md`; 3 topics × {NL-best, kw, hybrid}, 4
  judgments/pair, postcutoff judged on tesla1 after a quadro fairshare stall). **TOTAL DISSOCIATION:**
  metric leader = hybrid on all 3 topics; judge ranks hybrid LAST on all 3. CLINCHER — on post-cutoff
  the kw ToC covers 0.36 of the 2025 field vs NL's 0.21 (70% more) yet the judge picks NL 4-0. → the
  LLM judge is **prior-bound, not coverage-tracking**; it is NOT a valid arbiter on the niche/frontier
  regime this program targets (P1 circularity demonstrated cleanly). kw vs NL judge: tie on canonical
  ds (2-2), NL>kw on the niche topics. **Implication: the built-book test (Phase 2) is the only valid
  arbiter of the metric↔judge split.** Remaining: bandits completion, web-ON cells (optional — judge
  is shown invalid for niche, so metrics + built book carry the weight).

## Proposed


- **Cross-view agreement** (from handoff): require miner-pattern support inside geometric
  components — folds into Track 1 arbitration (P1 settled the direction: corroboration, not
  arbitration-lead).

## Rejected / closed

- z-ratio as written (radius-Δ/count-σ AND count-ratio Fightin' Words forms) — twice refuted
  on measurement; planted cores scored below fg-dense controls. Salvage = LCB (kept) and B.
- Soft multiplicative `g^a` habitability weight — hurts within scope (0.84 → 0.78); resolved
  by A: adopted as a binary veto in-scope (works, rank-all 13), refuted as a scope
  REPLACEMENT (best unscoped 8,708 ≈ full statistic's 8,860 vs scoped 13).
- Level-set components for small-plant RETRIEVAL — recall@top-3 = 0 across 48 trials;
  RESOLVED by P2: revived as a Track-2 change detector (differencing), still closed for
  single-snapshot retrieval.
- Further legacy greedy k-ball investment — 0.31 hit-rate, dominated; retained as baseline.
- Commute/effective-resistance geometry (P3b) — concept at chance (0.47–0.49) in every
  construction; flat bipartite spectrum = degeneracy regime. No place in the composite.
- Low-df keyword tail for the miner (P4) — 128.5 → 128.5, tail attribution 0.00; the
  champion's bound is corpus structure, not lattice resolution. C's "possible flip" refuted.
