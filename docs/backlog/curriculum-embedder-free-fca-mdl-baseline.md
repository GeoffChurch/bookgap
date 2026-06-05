# Embedder-free curriculum via FCA concept lattice + MDL selection

**The capability.** A purely symbolic, *embedder-free* concept-discovery + chapter-selection method:
build a Formal Concept Analysis lattice over the concept works, then select the chapter set by **MDL
(Krimp-style compression)** — canonical, knob-free (no `n`, no `τ`, no cohesion threshold), and it
chooses its own chapter count. Generalises the single "soft formal concept (FCA Galois closure)" of
[[curriculum-keyword-concept-seeding]] (the W2K/K2W fixed point) from one closed concept to the whole
lattice of stable concepts.

**Why keep it (the value proposition is OOD).** The default curriculum (GCCA multi-view consensus + merge)
leans on GIST/Gemma embeddings for *surface-invariance* (it maps every activation-function work to the same
neighbourhood regardless of which function it names). That advantage **evaporates on corpora that are
out-of-distribution for the embedder** — specialised notation, low-resource languages, novel symbolic /
formal / mathematical text. There, "the best *non-neural* method" is the right tool, and this is a working
one. **This is NOT a dead end** — it loses only the *in-distribution* last mile to GCCA+merge; its real test
is an OOD corpus, where it could win.

## Validated recipe (prototyped 2026-06-05; `fca_lattice_probe.py`)

```bash
# MDL (Krimp) embedder-free concept selection, with surface canonicalization
.conda/bin/python -m artifacts_scripts.curriculum.fca_lattice_probe \
  --concept-gap artifacts/book_gaps/<book>_concept/concept_gap.json --field 17 \
  --context concat --select mdl --canonicalize
```

Pipeline, in order (each step earned by an ablation — see findings):
1. **Formal context** = works × attributes. Attributes from the **concat_vocab** join {keyword terms ∪ cited
   references}, but **incidence MUST be lift-distinctive** (term over-represented vs the field by ≥ `--min-lift`),
   not raw presence.
2. **Closed concepts** (attribute concepts / closed itemsets, `|intent| ≥ 2`).
3. **MDL / Krimp selection**: keep each candidate pattern iff it lowers the two-part description length
   `L(D|CT) + L(CT)`. Mega-blobs (length-1 intents) carry no compression; redundant duplicates get ~zero
   usage → both auto-rejected. MDL **chooses the chapter count**.
4. **Optional embedder-free canonicalization** (`--canonicalize`): stem-merge (network/networks, snn/snns)
   + **extensional clump** (union-find merge of attributes whose extents have Jaccard ≥ `--clump-jaccard`)
   — the combinatorial substitute for the embedder's semantic merge (snn↔spiking-neural, mzi↔mach-zehnder).

## Findings (neuromorphic, field 17, N≈1899; the ablation ladder)

- **Lift incidence is mandatory.** Raw binary term-presence → the antichain/MDL collapse onto generic
  vocabulary (`accuracy, achieve, across, additionally`). Lift (over-rep vs field) → real anchors
  (`spiking`, `vcsel`, `reservoir`, `mach-zehnder`, `relu`).
- **Intensional stability is meaningful only mid-lattice.** It is degenerate (≈1.0) at *both* size extremes
  (singleton intents are trivially preserved), so it is **not** a good standalone ranker.
- **Selector matters; MDL is the right one.** max-**cohesion** antichain → niche fragments (25% coverage,
  *misses* the big spiking/ReLU topics); stability-weighted **set-cover** → 4 redundant mega-blobs. **MDL
  recovers the right ~15 mid-grain topics and picks the count** (neural-nets, relu, reservoir, snn, lasers,
  mach-zehnder, vertical-cavity, optical-feedback, phase-change, …).
- **The citation view does NOT contribute at the lattice level.** `concat ≡ terms` (refs never form closed
  multi-attribute intents at usable support — coupling is too sparse); pure-`refs` is coverage-capped (~30%
  object coverage). Contrast GCCA, where full-coupling *does* help — the difference is that CCA/GCCA use the
  *continuous* coupling features, FCA needs discrete co-occurring refs (rare).
- **The residual gap is combinatorial-semantic, not surface.** Even after canonicalization (904→721 attrs),
  MDL fragments the activation family across `{relu,sigmoid}`, `{relu,tanh}`, `{relu,leaky}`, `{relu,swish}`,
  `{relu,elu}` — *distinct co-occurrence patterns*, not surface variants. Closing this embedder-free requires
  agglomerating the *patterns* by attribute overlap — **which reinvents clustering**, and clustering in a
  continuous semantic space (GCCA) does it better on in-distribution data. Confirmed from three directions:
  the co-clustering dead-end, the CCA design rationale, and this probe.

**Verdict.** On in-distribution English CS, FCA+MDL(+canonicalization) recovers the right concepts but loses
the polished last mile to GCCA+merge (embedding surface-invariance is irreducible there). Banked as the
**best embedder-free baseline**, to be deployed/validated where embeddings are untrustworthy.

## To act on this later

- **The decisive experiment is an OOD corpus.** Find a domain where GIST/Gemma are weak (low-resource
  language; formal/math notation; a specialised symbolic field) *with* a ground-truth chapterization (e.g. a
  book ToC), and run FCA+MDL vs the embedding pipeline. This method's case is that it *wins* there.
- If continuing to chase the in-distribution last mile (lower priority): a pattern-level agglomeration of the
  MDL patterns by attribute-Jaccard — but note this is reinventing the clustering the embedder already does.
- A hybrid is possible: MDL proposes the candidate concepts / count, a light merge finalises — but that
  reintroduces an embedder or a knob, defeating the point.

## Pointers

- **Prototype:** `artifacts_scripts/curriculum/fca_lattice_probe.py` (`--select {mdl,setcover}`,
  `--context {terms,refs,concat}`, `--canonicalize`, `--min-lift`, `--clump-jaccard`). Self-documenting;
  throwaway-grade but the reference implementation of the recipe.
- **The default it is measured against:** GCCA multi-view consensus + coherence-merge — now the default in
  `run_concept_curriculum.py` (`--cluster gcca`). See `docs/backlog/curriculum-keyword-concept-seeding.md`
  ("GCCA PROMOTED…" section, 2026-06-05) for the promotion + the redundancy blemish this thread was spun off
  to address.
- **Single-concept ancestor:** the W2K/K2W soft Galois closure in [[curriculum-keyword-concept-seeding]].
