# OSM building-completeness via context-expectation — QUALIFIED: a fringe-of-mapped-fabric signal, dead in interiors (where the value is)

**Tested 2026-06-15** (gap-method domain-transfer spike; scripts
`artifacts_scripts/osm_spike/{fetch_osm.py, plant_recover.py, external_oracle.py, salvage_blockscale.py}`;
external oracle = Microsoft Global ML Building Footprints).

Context: brainstorming where the gap method transfers beyond articles/code. OSM completeness was the
**strongest non-code candidate** — a *forward* expectation-gap (a building structurally/contextually
implied by its surroundings but absent), with what looked like a clean external oracle (reference
footprints), open data, and an OOD-for-the-embedder bonus. The detector: a context-expectation model
("a building is expected here, given nearby roads / existing buildings / POIs").

**Expected:** the context model localizes genuinely-missing buildings, validated against an external
footprint dataset — i.e. it finds where OSM is incomplete.

**Actual — three stages, each deflating the last:**
- **Synthetic plant-and-recover** (random/MCAR deletions; Heidelberg): AUC **0.92**, beats a
  building-density baseline → looked strong.
- **External oracle, salt-and-pepper missingness** (Tirana; MS footprints): strict isolated-gap AUC
  **0.41 — worse than random.** The MCAR number did NOT transfer. Tirana's real missing buildings were
  scattered singletons (52/63 components size-1), invisible to a context model. (A 50 m grid +
  MS/OSM registration offset also inflated a misleading as-is 0.68.)
- **Salvage — block-scale holes + polygon-level matching**: the block-scale regime IS findable
  (Antananarivo, Lagos-Ikorodu: partial coverage with contiguous holes of 1k–2.9k footprints, vs
  Tirana's singletons). On the cleanest tile (**tana_north, 57% missing**): AUC **0.730** context vs
  **0.583** road-only vs **0.500** random; missing buildings carry **5×** the OSM-building context of
  empty space. Clean Heidelberg specificity (**0.871**, no manufactured gaps); polygon matching removed
  the grid artifact (τ-stable, no cliff). **But** it degrades to **random in deep interiors** of large
  holes (tana_south 0.557 — interiors have no nearby fabric, look identical to empty space) and under
  **saturation** where OSM≈external (Lagos-Ikorodu 0.525, road-density alone wins). Net:
  **MCAR 0.92 → real block-scale 0.73 best / ~0.55 typical.**

**Diagnosis:** the determinant is whether a hole **abuts already-mapped fabric**. Context-expectation
fires only at the **fringe** (missing buildings adjacent to existing OSM buildings → context exists to
predict from), and is structurally blind in the **interior** (no nearby fabric → indistinguishable from
empty space) and under **saturation** (OSM≈external → no contrast). The trap: the model's *competence*
region is exactly where the gap is **already visible to a human** (an edge hole next to mapped fabric),
and its *blind* region is exactly where the **value** is (whole unmapped interiors / informal
settlements). Worse, that high-value blind region is **owned by imagery extraction**
(RapiD / MapWith.AI / ML footprint models), which needs no map-context and already fills deep holes. So
as a *general* missing-building oracle it is refuted; it survives only as a low-leverage
"extend-from-the-mapped-edge" triage prior — where mappers can already see the gap.

**Salvage / what would change the verdict:** (1) fuse with imagery features — but then imagery
extraction is doing the work, not the gap method; (2) ship an explicit "frontier-completion" product
(prioritize tracing at the mapped edge) — narrow, and largely redundant with what mappers already see;
(3) a generative model that predicts fabric from **roads alone** could crack road-mapped interiors, but
that is no longer the context-density gap method. The durable, reusable finding: **context-expectation
completeness is a fringe-of-mapped-fabric signal (AUC ~0.73), mechanistically dead in interiors** — and
the **synthetic MCAR test (0.92) massively over-promised** vs real clustered missingness (0.55–0.73).
This is the same lesson the project keeps relearning: plant/synthetic evals over-state vs real holdout
(cf. the code temporal negatives; coverage-oracle-on-well-tested-packages). Sibling domain-transfer
result: [funding-waste-inverse-delta](funding-waste-inverse-delta.md) (failed for the opposite reason —
the two axes were positively *coupled*; here they dissociate but the useful regime is incumbent-owned).

Provenance: `artifacts_scripts/osm_spike/` (untracked exploratory; `salvage_blockscale.py` reproduces
the decisive block-scale test).
