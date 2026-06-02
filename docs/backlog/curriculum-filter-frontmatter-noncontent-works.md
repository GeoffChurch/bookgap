# Filter non-content works (front-matter / editorial / course boilerplate) at ingest

**Problem.** Color Image's curriculum has a chapter (ch0) that's a coherent cluster of **publisher
front-matter**: `series, departments, grad, oppenheim, alan, multidimensional, covers, courses,
digital_signal_processing, text, principles` — i.e. Oppenheim's Prentice-Hall *Signal Processing
Series* front-matter, course catalogs, editorial-board / "contents" pages. These are real corpus
records (book-chapter / front-matter entries) with no research content. They:
- survive the semantic KDE re-rank (DSP front-matter is *semantically near* image processing), and
- survive dynamic top-k (they form a real high-score cluster, not ~0-score padding),

so they cost specificity (gemma: "departments, grad … nonsensical/unrelated"). This is the
**boilerplate-cluster** debris mechanism, distinct from the padding debris dynamic-k fixes (see
[[curriculum-specificity-semantic-reweight]]). It's a **data-quality** issue, fixed upstream, not a
selection/display fix.

**Ideas:**
- **Title/type heuristics at ingest:** drop records whose title matches front-matter patterns
  (`front matter`, `editorial board`, `contents`, `series page`, `author index`, `subject index`,
  `call for papers`, `table of contents`) and/or OpenAlex `type` in {editorial, paratext, …} with
  empty/near-empty abstracts. Cheap, high-precision for the obvious cases.
- **Abstract-quality gate:** require a minimum real-abstract length / dictionary-word ratio; many
  front-matter records have null or boilerplate abstracts. Composes with the multilingual-noise
  observation from the Neuromorphic padding case.
- **OpenAlex signals:** `is_paratext` flag if present, or `type`/`type_crossref` filtering at
  `openalex_ingest`. Check whether the snapshot carries `is_paratext` — if so this is nearly free.

**Scope note:** decide whether to filter at ingest (cleans every downstream use, but re-snapshots
the scope) vs at selection time (cheaper, scoped to a run). Ingest is cleaner long-term. Eval via
the ToC-reconstruction panel — re-run Color Image and watch its specificity (currently the only
panel book still losing it after semantic-rerank + dynamic-k).
