# RUN QUEUE — the standing GPU pipeline (owner-ordered 2026-06-12)

**Purpose**: a fresh agent must be able to take over this pipeline cold from THIS doc + the
per-item pointers. Update it as items complete (append to §Readouts; flip statuses). The owner
has pre-authorized the whole chain — items fire AUTONOMOUSLY on the previous item's completion,
no approval gates unless marked.

**Hard rule**: ONE GPU job at a time (24GB; gemma 31b ≈ 19GB, goedel 32B ≈ 24GB). The chain is
strictly serial. CPU work (analysis, subagents) may run in parallel with any GPU job.

## Queue (status as of 2026-06-12 ~01:30)

### 0. ds3 book build — minimal-prescription arm  [ROUND 1 DONE — budget exhausted, see Readouts; ROUND 2 queued after Phase 1]
- Run dir: `~/src/learn-and-teach/outputs/2026-06-12/00-53-18/` (launched 00:53).
- What it is: 75-line handoff + harness, NO roster/phases/skeleton — the prescription-ablation
  probe. Engagement: `~/src/learn-and-teach/engagements/datascience-3/`.
- Done predicate: `engagements/datascience-3/done.sh` (digest 0 hard fails + ≥80pp + no stubs);
  auto_continue max 40 nudges; bail file `ERROR` in run dir.
- Monitoring: hands-off (the run IS the measurement; no content rescues; harness-level
  intervention only). Check script: `artifacts/lean_curriculum_experiment/scripts/check_ds3.sh`
  (10-min + 1-hr checks fired; further checks ad hoc).
- On completion: write a readout into §Readouts below (digest state; which structures gemma
  reinvented — planning/tournament/delegation; failure modes vs ds2; ToC vs the task_t3 audit
  gap list), then fire item 1. At 10-min check it had self-authored a status.md + one-shot ToC
  (no tournament), 0 subagents, 0 errors.

### 1. Curriculum-showdown Phase 1  [COMPLETE 2026-06-12 ~07:15 — independently verified; see Readouts + artifacts/gap_detection_eval/showdown/p1_results.md]
- The experiment: do algorithmically derived curricula beat agent ToCs, with the predicted
  INTERACTION (agent wins canonical topics; algorithmic wins niche/post-cutoff). Owner approved;
  web access for agent ToC authors is ALLOWED.
- Phase 0: COMPLETE (2026-06-12 ~02:05; worklog "Curriculum Showdown" section has the full
  numbers). Frozen config: LSA-64 seed42, KMeans k=11 chapters / 78 sections (pre-committed
  silhouette+share rule), register-filtered labels, exact min-forward-flow ordering over
  154,851 dated edges (`cache/oa/ds_showdown_refs/`). DS-slice metrics (algo vs ds2 ToC):
  coverage 0.993 vs 0.504 (CAVEAT: algo partitions the very corpus — structurally favored;
  the number to carry is "ds2 spans ~50% of field mass"); gap-closure 3/12 vs 0/12 (ds2=0 BY
  CONSTRUCTION — incumbent); prereq-violation share 0.444 vs 0.497 (vs 0.476 for ds2's
  partition optimally re-ordered — the ordering win is real but modest). Algo labels are raw
  term-tuples (chapter titles need the hybrid arm). Interpretive note: on the canonical topic
  the algo ToC reads as a RESEARCH-FIELD map (Ch1 SGD/nonconvex, Ch2 bandits, Ch3 random
  matrices), not undergrad pedagogy — consistent with the predicted agent edge there.
  Code: `curriculum_toc.py`, `toc_metrics.py` (+8 toy tests, suite 68 green).
- POST-CUTOFF WRINKLE (Phase 1 must handle): gemma 4 documented cutoff = **Jan 2025** (gemma 3
  = Aug 2024), so the Phase-0 window (≥2024-07) OVERLAPS the prior. At Phase 1 start: recount
  T10028 Topic Modeling (picked at 16,683 works ≥2024-07) with `from_publication_date:2025-02-01`;
  keep if ≥8K works, else widen to runner-up T10181 NLP or union. Record the re-cut on counts
  BEFORE pulling.
- Phase 1 spec (write its dispatch brief from this + Phase 0's final report): topics =
  {data science (canonical; corpus cached `cache/oa/t3_ds_book/`), causal inference (niche;
  T10845 sub-slice, ~10.3K works), the Phase-0-locked post-cutoff topic (pull its corpus then)}.
  Arms per topic: agent ToCs (gemma via pi, web on), algorithmic ToCs (FROZEN Phase-0 config),
  hybrid (algorithmic skeleton → agent polish). Scoring: provenance-blind randomized pairwise
  judging (gemma judge) + the objective metrics (field-mass coverage, gap-closure,
  prereq-violation count from dated citation flow) on EVERY candidate. Readout: per-regime
  Elo + metric table → owner. GPU: yes (gemma) — hence queued behind ds3.
- Standing program rules apply (see §Constraints).

### 1.5 ds3 ROUND 2 [RUNNING since ~07:25 — same run dir, fresh session, round-1 logs archived as *.r1]: relaunch `./engagements/datascience-3/run.sh hydra.run.dir=outputs/2026-06-12/00-53-18` from learn-and-teach root (fresh session, resume contract picks up the 8 chapters; second 40-nudge budget; hands-off).

### 1.55 Showdown R0 — round-trip E/D operators  [COMPLETE 2026-06-12 ~10:40 — report showdown/r0_roundtrip.md; agent parked post-score, dispatcher wrote report from banked results]
- Owner-approved round-trip program (extract ToC→keywords, derive keywords→ToC; M and A forms).
  Brief: docs/handoffs/2026-06-12-showdown-r0-roundtrip.md. Builds closure-based E_M (FCA-style,
  miner-backed) + dual-mode D_M; verifies M-side idempotence (the owner's design criterion);
  computes R_AM ("can mechanics improve the agent's own ToC?", incl. DS) + alignment-energy
  δ(T, R_MM(T)) for all 9 P1 candidates (intrinsic judge-free gradient test). Task #28.
  Its r0_datascience_agent_RAM_free.md feeds P1b's judge cell.

### 1.6 Showdown Phase 1b — keyword-vs-NL axis + detector-topic + R1 cells  [QUEUED — fires on ds3-r2 completion, before ds4]
- Owner-proposed: per topic, 3 new agent cells {kw-web-on, kw-web-off, NL-web-off} vs P1's
  banked NL-web-on. Keywords = top-40 register-filtered c-TF-IDF terms (frozen vectorizers, no
  titles); de-labeled audiences for kw cells; 1 run/cell (flag the K asymmetry vs P1 best-of-3).
- Scoring metrics-primary (coverage, violations) + term-anchoring (chapter vocab ∈ corpus
  top-200 vs outside). Readings: kw competitive web-on ⇒ corpus-in-book-out (no topic string —
  inverts the compiler's topic-string-in goal); kw>NL web-off on postcutoff ⇒ prior-retreat
  mechanism confirmed. Reuse P1 mini-run mechanics (briefs/driver/ledger pattern).
- PLUS the 4th, DETECTOR-SELECTED topic (owner asked; closes the detector→topic→showdown loop
  none of P1's topics ran): pre-committed rule = highest scope-share among ds2's audited
  curriculum absences with ≥5K works ⇒ BANDITS/sequential decision-making (scope-share 0.124,
  7,580 works in the t3 audit). Pull/clean per t3 recipe → algo+agent+hybrid ToCs + kw cells +
  full metrics. P2's build target (causal vs bandits) is then decided on results, not guess.
- PLUS R1 cells folded in (owner 2026-06-12): E_A extractions (agent-DS, algo-DS), R_AA drift
  loop ×3 on agent-DS, blind judge of agent-DS vs R0's R_AM round-trip (4 judgments, P1
  protocol). Requires R0 artifacts (item 1.55). ~16 runs ≈ 2.5h GPU. Task #27.

### 2. ds4 book build — rich-single-doc arm  [QUEUED — fires on Phase 1b completion]
- Engagement READY + verified: `~/src/learn-and-teach/engagements/datascience-4/` — ds2's full
  instruction set (phases, ToC tournament, role contracts, skeleton, pitfalls) in ONE 213-line
  document, big-picture stance, NO roster (executor writes helper briefs from in-document
  contracts).
- Launch: `cd ~/src/learn-and-teach && ./engagements/datascience-4/run.sh` (background) + arm
  10-min/1-hr checks (reuse check_ds3.sh with the new run dir). Hands-off observation, same as ds3.
- Comparisons: vs ds3 = instruction richness (same stance, same no-roster); vs ds2 = roster
  removal (+stance change). On completion: readout into §Readouts, fire item 3.

### 3. Curriculum-showdown Phase 2 — the niche textbooks  [QUEUED — fires on ds4 completion]
- Build REAL books from the per-regime winning ToCs of Phase 1 — at minimum the niche topic's
  algorithmic-vs-agent pair (the interaction's money cell). One GPU night each.
- Mechanism (frozen-ToC injection): create a fresh run dir, pre-seed
  `build/state/toc.md` (+ a status.md noting "ToC frozen externally") with the winning ToC,
  then launch the engagement's run.sh with `hydra.run.dir=<that dir>` — the resume contract
  ("continue from what exists; do not redo the ToC") skips ToC generation. Use the ds4
  engagement as the base harness unless the owner says otherwise.
- Score the FINISHED books with the Track 3 audit machinery (fg = the book, bg = the topic
  slice): end-to-end gap-closure is the outcome. On completion: readout, fire item 4.

### 4. Lean confirmatory sweep — RESUME  [QUEUED — fires on Phase 2 completion]
- State: PAUSED 2026-06-12 ~00:30 at 337/1875 rows (draw 0: 326/625), zero data loss.
  run_id `confirm_v3_goedel32b`; results:
  `artifacts/lean_curriculum_experiment/confirm_v3_goedel32b/results.jsonl`.
- Resume: relaunch `artifacts/lean_curriculum_experiment/scripts/run_confirm_v3.sh` (durable
  copy; original /tmp copy may be gone) in background + the watchdog
  `scripts/watch_confirm_v3.sh`. Resume-skip replays nothing (banked (arm,target,draw) rows of
  the same run_id are skipped). After 32B finishes (5 arms × 125 targets × 3 draws), the script
  runs gemma control+t1 automatically, then touches /tmp/confirm_v3_done.
- Interim analysis anytime: `python3 curriculum_lean/interim_readout.py` (exploratory label).
  At pause: t1_shuffled +9.2% vs control (n=65 complete blocks, McNemar p=.031); t1 +6.2%
  (p=.22); irrelevant placebo +3.1%. Pace ~2.8 min/row ⇒ ~3 days of GPU for the remainder.
- Pause/instrumentation history: `docs/preregistration-curriculum-context-v2.md` §10.
  Anomaly auditor: `scripts/check_confirm_anomaly.py` (tokens=0 rows; base rate ~0.3%).

### 5. Sweep analysis + the Markov book — THE END  [QUEUED — fires on sweep completion]
- Pre-registered analysis per `docs/preregistration-curriculum-context-v2.md` (§5 metrics:
  audited pass@3, exact McNemar, H1–H5; ingest ONLY run_ids confirm_v3_goedel32b /
  confirm_v3_gemma_off). Report to owner.
- Then launch the markov-categories book (owner's standing order, explicitly
  deferred-until-the-end): `~/src/learn-and-teach/engagements/markov-categories/` (built by the
  compiler WITH a roster + dual-track analytic||synthetic spec). OPEN QUESTION for the owner at
  trigger time: keep its roster, or port it to the no-roster style per the new philosophy.

## Constraints (program standing rules — apply to every item)
- TRANSCRIPT HYGIENE in all subagents: titles/top-terms/counts/numbers/code only; never corpus
  body text (policy false-positives have killed agents).
- Never signal processes you didn't start; anchor pgrep patterns (`[n]ame`) — the self-match
  loop has wedged this box 3+ times. One GPU job at a time; CPU subagents cap BLAS/n_jobs ≤ 8.
- bookgap: NO commits, NO tracked-file edits (note: curriculum_lean/experiment.py +
  goedel_prover.py carry uncommitted instrumentation edits — do not revert). learn-and-teach:
  apparatus/book_digest.py carries the uncommitted overfull/width gates — do not revert; owner
  chose to leave them uncommitted. OpenAlex: polite pool + page caching, always.
- Books: observation is hands-off (no content rescues — the runs are measurements). Harness
  intervention only on wedge/crash.
- Memory dir (`~/.claude/projects/-home-gchurchill-src-bookgen/memory/`) has the deep context:
  active-run-queue.md points here; gate-arms-race.md, datascience-textbook-run.md,
  orchestrator-context-overflow.md are the relevant war stories.

## Readouts (append per completion)

### Showdown Phase 1 (2026-06-12 ~02:13–07:15, two agents, zero collision, independently verified)
- Post-cutoff slice locked on counts: T10028 Topic Modeling, 2025-02-01→open, 9,944 works
  (only 697 in-slice citation edges — ordering leaned on median-year tie-break, on record).
- BLIND JUDGE (gemma, 36/36 valid): agent sweeps every pair on every topic 4–0; hybrid beats
  algo everywhere. No interaction at this lens. Blinding was porous (algo's unpolished register
  recognizable); on post-cutoff the judge used its pre-cutoff topic prior to penalize what the
  2025 corpus actually contains (RAG/agents/compression) — judge≈generator circularity, on record.
- METRICS: the predicted interaction as a monotone gradient — agent coverage of real field mass
  0.50→0.31→0.12 (canonical→niche→post-cutoff), violations 0.47→0.58; algo holds ~0.99 coverage
  (0.28 violations post-cutoff). DS gap-closure: algo 3/12, hybrid 1/12, agent 0/12.
- 20/21 judge-vs-metric cells DISAGREE. HYBRID is the only arm strong under both lenses
  (judge-preferred over algo 4–0 ×3; recovers ~half the agent coverage deficit) → P2 amended:
  build hybrid-vs-agent books on causal.
- Continuation agent independently re-derived blinding (36/36), re-tallied all verdicts,
  recomputed metrics exactly, found zero provenance leaks. Suite 74 green. My ops errors tonight
  (instrument, not pipeline): etime misread killed one healthy mini (retried clean), 3 watchdog
  false alarms from harness quote-mangling of inline -newermt (fix: script-file watchdogs +
  arm-time self-test).

### ds3 round 1 (2026-06-12 00:53–02:14, exit: 40/40 auto-continue budget EXHAUSTED, book NOT done)
- Produced: 8/10 planned chapters drafted at real depth (ch0–ch7, 13–22KB LaTeX each, 131KB
  total), one-shot ToC (self-authored status.md with checklist — held stable all run), a
  self-written debugging tool (find_unclosed.py). Ran the digest oracle 41× (≈1/session) — the
  run-the-verifier discipline HELD with only prompt-level instruction.
- Failed: master book.tex never created → no book.pdf; chapters 9–10 undrafted; ZERO
  delegation (despite tool + context warning); ZERO own tectonic invocations (never
  compile-verified its LaTeX); path discipline collapsed — stray dirs `` ` ``build, 「build,
  C:\Users\Admin\build in the run-dir root; some figure PDFs written into sections/.
- Burn pattern: first session ~60 min (the drafting), then nudged sessions shrinking to ~2 min
  each (2.0MB session jsonl, single session continued 40× — consistent with context saturation).
- Disposition: round 2 QUEUED after showdown Phase 1 (fresh-session resume vs the file state —
  the harness's designed recovery; second 40 budget; NOT a content rescue). Then ds4.
