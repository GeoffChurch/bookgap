# Keyword-concept seeding: a soft target-works vector that replaces the embedding-ball gap

**Origin (conversation 2026-06-03).** The embedding-ball gap (`radius = 0.75 × dist-to-nearest-book`,
articles in the exclusive ball) is **empty for any non-isolated book** — Koehn SMT and Jurafsky SLP both
return `gap_size 0`, because their territory is densely covered. So today the method only fires on
isolated/niche books. A **keyword-concept** neighborhood exists for *every* book and gives a soft target-
works vector to seed the curriculum, decoupling seeding from embedding-isolation.

## The mechanism (reuses the validated nonlinear iteration)

From a single seed text (e.g. a book's 25-chapter ToC), iterate `seed → W2K → K2W → W2K → …` to a fixed
point (`gap_keyword_iteration.py`; converges in ~5–7 steps, seed-dependent coherent basins):
- **W2K** (soft `intent`): the lift distribution `log[P(t|seed) / P(t|field)]` — keywords that "best
  select the seed and exclude everything else." Single-text foreground is fine (it's the existing
  `w2k({seed})` seed step); phrase **vocabulary** is mined over the candidate pool and the seed is
  *parsed* with it, so single-text phrase mining isn't degenerate.
- **K2W** (soft `extent`): score works by how well they match the lift keywords (distinctive ⟹ rare ⟹
  short inverted-index postings ⟹ cheap, even at field scale; only the field DF vector needs caching).
- The **fixed point is a soft formal concept** (FCA): a closed (extent, intent) pair; the iteration *is*
  the Galois closure, and the nonlinearity (re-deriving lift each step) is why it converges where linear
  power iteration collapses. Robustness ↔ FCA **concept stability**.

**It IS the target-works weight vector.** The fixed-point K2W/responsibility scores over works are a soft
weight vector that drops in exactly where the embedding ball's hard `target_work_ids` set is consumed —
the bidirectional-PageRank **personalization**. Consequences: (a) **no empty-set failure** (a concept
always has support — rescues popular books); (b) **more signal** than a 0/1 ball (graded
"how seed-distinctive is this work" → personalization), all downstream (clustering → keywords) unchanged.

## All work types downstream (the article-only split is a gap-detection artifact)

The books-vs-articles `FilterPair` split is load-bearing **only** in the embedding-ball detector (books =
the *covering* set tested against). Everywhere else — the concept, the target-works vector, clustering,
keywords — should span **book + article + book-chapter + preprint** (a curriculum should be able to anchor
a chapter on a foundational book or survey). Future ingest: **reviews/surveys + dissertations** are ideal
curriculum material and worth pulling into scope. Paratext (editorials, "special issue" notices) stays
suppressed via the ∅-sink regardless of type.

## Spin-off — book-deficit-lift gap detection (maybe better than k-balls)

Once the concept weight vector spans all types, define a concept's **book-deficit** as the lift of the
`book` *attribute*: `P(book | concept) / P(book | field)`. A concept with lift **< 1** (books under-
represented vs the field base rate) is a coherent topic with active research but no book covering it —
*the* definition of a gap, now **graded and topic-coherent** instead of geometric + binary. Gap detection
becomes: enumerate concepts (keyword fixed points from many seeds), rank by book-lift; most-negative =
biggest book-gaps.

Why it could beat the k-ball:
- Measures the target directly ("is this *topic* under-covered by books?") vs the k-ball's proxy ("is this
  region embedding-isolated from books?").
- Graded (a ratio) vs the 0/1 book-existence filter.
- Topic-coherent and **not isolation-bound** — can flag a book-poor topic that isn't embedding-isolated (a
  k-ball blind spot) and reject a falsely-isolated niche that actually has a recent book.
- **Unifies gap detection into the lift/concept framework** — "find a concept where the `book` attribute
  has negative lift," the same machinery + boundary-background normalization (vs the field base rate)
  used for keywords, with work-type as the attribute.

Subtlety: books are globally rare (~59k books vs millions of articles in field 17), so the ratio **must**
be normalized against the field base rate — the lift form handles this; an unnormalized "low ratio" would
flag every concept.

## Plan

1. Adapt `gap_keyword_iteration.py` to seed from **external book text** (ToC) with a **field DF
   background** and inverted-index K2W over the field (distinctive-term postings only), all work types.
2. Fixed-point soft weight vector over works → normalize → bidirectional-PageRank personalization → Track A
   (product-of-experts + soft + RELEVANCE) → cluster.
3. Compare to the book's real ToC (Track B) + the fair judge — an *interpretable* "did the engine
   reconstruct a recognizable curriculum?" test. First target: **Jurafsky SLP** (25 ch).
4. (Later) the book-deficit-lift gap detector as a k-ball alternative — its own eval vs the validated
   gap-books.

## Follow-ups (deferred)

- **Basin-level paratext drop (the "corrected #1").** Per-work concept-distinctiveness was REFUTED for
  paratext (2026-06-03): all three forms (∅-leak/peakedness/maxresp over the basins) measure
  *concentration in one basin*, but paratext is a **coherent generic basin** (the "Introduction" basin:
  `proceedings/annual_meeting/association`) → concentrated → reads distinctive → uncatchable at the work
  level. Worse, they mis-weight: ∅-leak down-weights *surveys* (Goldberg's NN-for-NLP book);
  peakedness/maxresp down-weight cross-cutting works (coref+entity) and UP-weight off-topic-but-
  concentrated works (vision transformers sitting in the Transformers basin via shared `self-attention`).
  The defect is at the **basin level**: drop a basin that is **keyword-coherent but content-scattered**.
  - **Embedding-incoherence** (preferred): the paratext basin's works span all topics → low internal
    embedding cohesion, despite venue-keyword coherence. Recency-neutral.
  - **Citation-incoherence** (viable, but recency-confounded): a paratext basin's works barely cite each
    other; but a *recent frontier* basin (RLHF/RAG) is *also* citation-sparse (works too new to be
    cited) → false-flagged as paratext. So citation-incoherence alone penalizes the frontier we just
    preserved. Clean combination: **embedding-incoherent = paratext** vs **embedding-coherent yet
    citation-sparse = frontier**. Use embedding-incoherence as the single signal; citation-incoherence
    only age-normalized or as the second axis.
  - Defer until after seeing #2 (the coherence cut) alone — #2 already collapses the 3 paratext clusters
    →~1 (splitting paratext yields keyword-redundant children → stop), so the residual paratext may be
    small enough not to need the basin drop.

## #2 — coherence-merge (JS) status (2026-06-03)

Built `merge_redundant_topics.py`: keyword-JS between topics, **complete-linkage** agglomerative merge
(single-linkage chains through the paratext hub; complete-linkage requires mutual redundancy), τ via
`--tau` or `--permutation-null` (95th-pctile JS between two random halves of the same topic). JS as the
canonical density-invariant effect size (a likelihood-ratio/G-test would re-introduce density bias).

On Jurafsky decoupled (25 topics): JS cleanly separates redundant (paratext 0.02–0.15, speech 0.26–0.35)
from distinct (median 0.887). **But #2 ALONE has no clean operating point:**
- τ=0.3: 25→21, merges only genuine redundancy (paratext 3→1, ASR 2→1, TTS 2→1), no content lost — but
  *under*-merges (speech dupes remain).
- permutation-null τ=0.434 (same-concept-half JS mean 0.223, 95th 0.434): 25→18 but **over-merges —
  LLMs (`fine_tuning,plms`) absorbed into the paratext blob**.
- **Root cause:** the paratext basin is a generic *hub* — its `natural_language_processing/language` terms
  give it JS≈0.4 with content topics (LLMs), which sits *below* the same-concept-half null (0.434). So
  the hub's spurious cross-concept similarity is indistinguishable from real same-concept redundancy; no
  τ separates "merge dupes" from "absorb content into the hub."
- **Conclusion: the corrected #1 (drop the paratext basin) is a PREREQUISITE for #2, not a complement.**
  With the hub removed, the permutation-null τ should be clean and can fully collapse the speech
  redundancy without endangering content. → un-defer #1 (embedding-incoherent-basin drop) next.

## Corrected #1 (basin-incoherence drop) — FAILED + re-diagnosis (2026-06-03)

`gap_keyword_concept.py --drop-incoherent-basins` (compare each basin's internal embedding cohesion to
a random same-size null) — **dropped 0 basins**. ALL 25 basins scored coh 0.87–0.92 vs null ~0.84
(cohesion barely discriminates in the tight NLP region); the "Introduction"/paratext basin is coh=0.888,
**mid-pack, not an outlier**. The premise (paratext = embedding-incoherent) is wrong: proceedings-style
boilerplate embeds *coherently*.

**RE-DIAGNOSIS (inspected the works):** the "paratext" topics are NOT proceedings-volumes — their works
have real abstracts (mean ~270 ch, 0% empty) and are **diverse legitimate recent NLP papers**
(summarization, cross-lingual FT, entity typing, NLI, distillation, parsing datasets). They cluster by
shared **generic** vocabulary (`natural_language_processing/joint/papers/tasks/venue`), forming a
**"miscellaneous recent NLP" cluster with a generic label** — the *vague-chapter* defect, not paratext.
The embedder *groups* them (generic framing embeds similarly) → embedding-cohesion can't flag it.

**The keyword-native fix is LABEL GENERICNESS, not incoherence.** A generic/miscellaneous cluster's
distinctive keywords have **high cross-chapter document frequency / low concept-internal IDF**
(`natural_language_processing` is in every chapter; `wer`/`word2vec` in one). Detect by the cluster's
top-keyword specificity *within the concept* (curriculum-marginal lift / concept-internal IDF) — reuses
the field DF, needs **no embedder** (which actively mis-groups these by framing). Vindicates the
"do we need embeddings?" thread: here keyword statistics are both necessary (embedder fails) and
sufficient. NEXT (if pursued): a label-genericness filter/penalty at the cluster level, or fold
concept-internal-IDF into the clustering so generic-framing works don't co-cluster.

## (a) genericness metric + (b) keyword-TF-IDF clustering (2026-06-03)

**(a) `topic_genericness.py`** — per-topic mean field-IDF of its keywords (embedder-free; reuses the
field DF). Curriculum-level metric + diagnostic. On Jurafsky-decoupled: mean 5.33; flags the genuinely-
generic *single-word* boilerplate topics (`machine_learning/accuracy/detection`, IDF 3.4) but NOT the
venue-phrase misc topics (`computational_linguistics/joint_conference` are specific *phrases* → high
IDF). So IDF catches generic single words, not venue phrases.

**(b) `cluster_concept_by_keywords.py`** — cluster the concept's works by **concept-internal TF-IDF**
keyword vectors (generic framing terms → IDF≈0 → down-weighted), agglomerative average-linkage on
cosine. Result on Jurafsky:
- **WIN — structure + specificity:** the 18 big clusters (347–1210 works) are clean Jurafsky chapters
  *including the frontier* (rlhf/rag/dpo, bert, self-attention, word2vec). Genericness **mean-IDF 8.74
  vs 5.33** (min 7.11 vs 3.37) — the generic/boilerplate blobs **dissolve**. Transparent, **no neural
  embedder** — keywords carry the structure *better* than the embedder on specificity.
- **LOSS — surface-form fragmentation:** the tail (6 clusters of 4–38 works) segregates by **language**
  (Indonesian `akurasi/ejaan`, Spanish `aprendizaje_profundo`, French `resultats`) + tiny niches; and
  the **venue/paratext cluster persists** (proceedings phrases are high-IDF, not down-weighted).

**Resolution of the "do we need the embedder?" thread:** for the **core structure + specificity**, no —
keyword-TF-IDF (concept-internal) is transparent and *better* (dissolves the generic blobs the embedder
manufactures). The embedder's **only** residual edge is **robustness to surface form** (it groups same-
content works across languages/phrasings; keyword-clustering fragments them). **Synthesis = hybrid:
keyword-TF-IDF drives structure + specificity; embeddings only merge the cross-language/paraphrase
fragments** (e.g. an embedding pass that folds the foreign-language micro-clusters into their content
chapters). The keyword-coclustering dead-end (refuted) was global/field-IDF judged on embedding-
silhouette; **concept-internal TF-IDF on the *selected* concept is a different, better-behaved regime**
(this is the scoped re-test that beats it on specificity). Practical balance fix remains #2's complete-
linkage JS merge at safe τ; (b) is the structural alternative whose remaining gap is robustness.

## Follow-ups (CCA hybrid)

- **Ablate the "extra stuff" in the RCCA hybrid.** The canonical recipe (concept-TF-IDF → LSA → ridge-
  regularized CCA with the embedding → cluster the shared subspace) has several moving parts that may be
  removable or improvable: (a) is the **LSA** reduction needed, or does sparse/kernel CCA on raw TF-IDF
  do as well? (b) how sensitive is it to the **CCA ridge λ** (sweep)? (c) **consensus vs E-only vs
  ρ-weighted** canonical coordinates for the clustering representation? (d) **k-selection** (Bartlett vs
  elbow vs fixed)? (e) agglomerative-cosine vs spherical-kmeans on the shared subspace? Goal: the
  minimal version that keeps the win (keyword-specificity + language-robustness).
- **Linear-CCA caveat → a purely LOCAL alternative may be better.** CCA collapses **global** subspaces —
  one "language/disagreement" direction-set for the *whole* concept. But the embedder↔TF-IDF
  disagreement is plausibly **local** (language matters in some regions, not others; a direction that's
  nuisance for one chapter may be signal for another). A global linear subspace can't capture that.
  Local/colleague-proof variants where this might shine: **local/regularized CCA per neighborhood**,
  **manifold alignment**, or — the **sample-side dual** — **cluster-merging**, which is *inherently
  local* (merge *this* foreign-language fragment into *this* content chapter via E-similarity, no global
  direction removed). The merge-dual is the cheap local counterpart to the global CCA; worth A/B-ing once
  the global CCA is in, especially if CCA's global collapse over-removes signal somewhere.

## CCA hybrid — the synthesis, VALIDATED (2026-06-03)

`cluster_concept_cca.py`: ridge-regularized CCA between the GIST embedding (E) and the concept-internal
TF-IDF→LSA (T); keep the top-k high-canonical-correlation directions (content agreement), collapse the
low-ρ tail (= surface/language disagreement — T-specific, no embedding correlate since the embedder is
language-robust); cluster the ρ-weighted consensus canonical coordinates. This is the user's
"GEP to detect+collapse the directions where the embedder and TF-IDF disagree" — CCA is that GEP.

Three-way on Jurafsky (genericness = mean field-IDF of topic keywords; higher = more specific):

| approach | mean-IDF | foreign-label topics | tiny (<20) | character |
|---|---|---|---|---|
| embedder (decoupled) | 5.33 | 0 | 0 | robust but **generic blobs** |
| keyword-TF-IDF (b) | 8.74 | 5 | 6 | specific but **language-fragmented** |
| **CCA hybrid** | **8.71** | **1** | **2** | **specific AND robust** |

CCA keeps (b)'s specificity (8.71≈8.74; generic blobs gone) AND recovers the embedder's
language-robustness (foreign fragments 5→1, tiny 6→2) for −0.03 IDF. Cleaner separation than (b):
ASR vs Phonetics/MFCC split, RLHF vs RAG split, clean text-classification (logistic/naive-bayes/RF/kNN)
instead of a generic-ML blob; all frontier chapters present. **The ρ spectrum declines smoothly
(0.77→0.30, no sharp elbow)** — content/agreement is spread over ~30 directions; language/surface is a
*high-dimensional low-ρ residual* (the ~170 collapsed dims), which is why removing it cleanly de-
fragments without losing content. **Residual: the paratext cluster persists** — proceedings titles
embed coherently AND carry venue keywords, so BOTH views agree → CCA keeps it (it's a genuine both-views
cluster, not a disagreement artifact → needs a content-hygiene/work-type filter, not a representation fix).

**Resolution: the hybrid is a CCA shared-subspace** (keywords for structure+specificity, embedding for
surface/language robustness). The user's GEP/collapse-disagreement framing = CCA, canonically. Open:
the ablations + the local (non-global-subspace) alternative are backlogged above; the paratext needs a
separate hygiene filter.

## Generality sweep (de-incant experimentation, 2026-06-03)

`experiment_concept_generality.sh`: concept-seed + CCA + keyword-TF-IDF cluster, across 3 more field-17
books (neuromorphic 13ch, color_image 24ch, mcs 10ch). **Keyword-concept seeding works on niche books
too** (not just popular ones). genericness (mean field-IDF; CCA vs kwcluster) / tiny-clusters:
neuromorphic 9.70/4 vs 9.88/8 · color_image 8.94/4 vs 9.09/5 · mcs 8.69/1 vs 8.83/5. Patterns: **both
keyword methods are highly specific on every book** (8.7–9.9 — generic-blob problem dissolved
universally); **CCA consistently has fewer tiny fragments** (for −0.15 IDF). Foreign-language advantage
is topic-dependent (big for NLP, ~0 for English-dominant niche), but CCA's general de-fragmentation
holds. → **CCA generalizes; de-incant leans CCA.** Remaining decider: the LM judge (CCA clustering vs
embedding-dendrogram clustering, same concept seeding) — the external signal.

## Follow-up — productionize the entry-point

`run_concept_curriculum.py` (built 2026-06-03) is a **prototype orchestrator**: it subprocess-chains
`gap_keyword_concept` → `cluster_concept_cca` (+citation ordering) → `render_judge_summary`, **reloading
the cloud per step** (tolerable — the index + citation graph are joblib-cached, but wasteful). To
productionize:
- Refactor concept-seeding / CCA-clustering / ordering into **importable functions** so it's one cloud
  load (and foldable into the main curriculum pipeline as a `--seed-mode keyword-concept` option, peer
  to the embedding-ball gap).
- Fix the **embedding-path output-dir doubling quirk** (`book_curriculum` prepends `artifacts/curriculum/`).
- **Flip the default** to this path once the multi-book LM judge confirms (judge is 1 book so far) and
  the paratext hygiene filter lands.
- Carry the chapter **ordering** natively (currently citation-balance net-cited-first inside
  `cluster_concept_cca`; the embedding path uses the richer frontier-adjacency — unify).

## Citation-view CCA (embedder-free attempt) + the discard-clash motivation (2026-06-03)

`cluster_concept_cca --view {embedding,citation,both}` on jurafsky:
- **citation–keyword are largely INDEPENDENT**: top canonical correlation **ρ=0.229** (vs embedding's
  0.77), declining slowly → CCA kept only **k=6** dims. (The embedder is text-derived → redundant with
  keywords → large shared subspace; citations are structural → tiny shared subspace.) Confirms
  "independence ⇒ much smaller shared subspace," strongly.
- **Consequence — resolution loss**: clustering 25 chapters on 6 dims → lumpy (sizes 1044→60); the LM
  judge prefers **embedding on all 4 dims** (citation loses 4/4). `both` ≈ embedding (ρ 0.75 — citation
  is redundant-given-embedding for the *agreement* signal, adds nothing).
- **So pure-citation-CCA FAILS** (the transparent-pipeline-via-CCA hope), and it pinpoints WHY: CCA
  *retains* the shared subspace, which is too small for an independent view. **The right tool for an
  independent view is discard-the-clash, not retain-the-shared** (confirmed empirically).

**Discard-clash route (the motivated next experiment, via `denoisers.py`):** don't retain the 6 shared
dims — instead KEEP the keyword content and REMOVE only the few dominant directions where keywords vary
but citations don't (= language/surface). That's the variance-ratio GEP `max wᵀΣ_keyword w / wᵀΣ_citation w`,
top-k removed — i.e. Anti-LDA / contrastive-PCA, **not** CCA's other end (CCA's low-ρ is *uncorrelated*,
which mixes language with citation-independent real content). Reuse `denoisers.py`: `_solve_eigh_robustly(A,B)`
(regularized generalized eigensolver), the scatter matrices, and **RNOJD** (joint-diagonalize Σ_keyword
& Σ_citation → rank directions by variance ratio = where the views clash). Also note: CCA *whitens*
(amplifies noise on the sparse citation view) → for citation, **PLS (covariance, no whitening) is the
safer pairing** if revisiting CCA. Eval: genericness + foreign-fragments + judge, same harness.

### Discard-clash RESULT — REFUTED, same root cause as citation-CCA (2026-06-03)

Built it (`cluster_concept_cca --method discardclash --discard-k 6`): regress keyword-LSA `T` on citation
features `C`, PCA the residual `T_res`, project `T` orthogonal to the top-k residual directions (the
claimed "language clash"). On jurafsky: genericness **9.35** (highest of any method — labels look
specific) but **n_tiny(<20)=7** (vs CCA 2, embedding 0; sizes 2067→2, several garbled chapters), foreign
**2** (Indonesian T7 survived; ACL paratext duplicated across T16/T24), and the **LM judge prefers
embedding on all 4 dims (discard-clash loses 4/4)** — identical to pure-citation-CCA.

**Why (decisive diagnostic, logged):** **citation R²=0.040** — citations linearly explain only 4% of
keyword variance (exactly ρ²≈0.05 from ρ=0.229). So `T_res ≈ T`, and the **top-k residual-vs-content
subspace overlap = 0.997**: the 6 directions we removed as "language" are 99.7% identical to `T`'s own
top-6 *principal content* axes. Discard-clash didn't sand off a language clash — **it amputated the
biggest content directions** (the dominant topics), which is why it fragmented and lost the judge. The
premise ("residual citations can't explain ≈ language") is false *because citations explain ~nothing*.

**Unified conclusion — both ends of the citation×keyword GEP fail for ONE reason (independence):**
CCA-*retain* keeps a 6-dim shared subspace (too small → lumpy, judge 4/4 loss); discard-clash-*remove*
strips the top content axes (residual≈content → judge 4/4 loss). There is **no usable clash subspace
because citations share NEITHER language NOR content with keywords** (R²=0.04). The embedder works as the
2nd view precisely because it is **text-derived → it shares the language subspace with keywords**, so
CCA's agreement=content / disagreement=language is meaningful; citations cannot triangulate language out
of keywords at all.

**Verdict on the symmetric cherry-pick (JIVE/GFA, "include a good citation subspace + remove a bad
keyword subspace"):** the *framework* is canonical and sound (not fumes) — but the obstacle **bites
hard** for the citation pairing. A symmetric decomposition can only excise the nuisance (language) if the
2nd view **overlaps the nuisance subspace**; citations overlap neither subspace (independence), so there
is nothing to cherry-pick. Salvage paths if the symmetric idea is revisited: (a) a 2nd view that *does*
share language but not content — e.g. a deliberately **shallow lexical embedding** (char-n-gram / surface
LM) as the language-sharing view, with the semantic embedder as the content-sharing view (two-sided JIVE);
(b) drop citations as a *denoising* view entirely and keep them only for **ordering** (already validated:
`--decouple-selection`). The language-squashing job stays with the (text-derived) embedder via CCA.

### CORRECTION (2026-06-03): the "citation⊥keyword" finding was a STARVED-SUBGRAPH artifact

`citation_spectrum_probe.py` re-measured the overlap on the *full* reference list instead of the internal
subgraph. The internal subgraph we'd been using (`G[rows][:,rows]`) has **9 581 edges over 9 886 works ≈
1 internal edge/work** — of ~8 in-field references per work, ~7 leave the concept. The Ruzicka+CSLS
feature on that is **near-identity** (erank 9327 ≈ N — a *pathological* high rank: structurelessness, not
richness), which is the real reason `R²=0.04` / `ρ1=0.229`. **It was noise, not independence.**

Effective rank (entropy erank; participation ratio (Σλ)²/Σλ²; stable rank Σλ/λmax) and keyword overlap,
jurafsky concept, N=9886:

| citation view | erank | partic | stable | kw R² | ρ1 | ρ̄(10) |
|---|---|---|---|---|---|---|
| CURRENT (Ruzicka, internal) | 9327 | 8973 | 3471 | 0.040 | 0.536 | 0.415 |
| internal directed | 1451 | 182 | 36 | 0.077 | 0.758 | 0.639 |
| internal symmetrized (G+Gᵀ) | 2365 | 272 | 43 | 0.076 | 0.785 | 0.659 |
| **full coupling (out-cites)** | **5544** | **1272** | **103** | **0.154** | **0.775** | **0.686** |
| full co-citation (in-cites) | 3798 | 121 | 20 | 0.033 | 0.540 | 0.426 |
| full concat [out‖in] | 5886 | 442 | 41 | 0.131 | 0.758 | 0.675 |

**What changes:**
1. **Use the FULL out-citation profile (bibliographic coupling), not the internal subgraph.** It
   **quadruples** keyword overlap (0.040→0.154) and has *real* structure (erank 5544, participation 1272 ≫
   25 chapters). ρ1=0.775, ρ̄(10)=0.686 — there IS a real ~10–20-dim shared subspace, so the earlier
   "shared subspace too small (k=6)" was also an artifact.
2. **Out-only beats symmetrize and concat.** Adding in-citations *dilutes* (concat 0.131 < out 0.154)
   because **co-citation is nearly signal-free here (R²=0.033) and recency-biased** (recent works aren't
   cited yet). The recency-clean view (out-cites, age-neutral) is ALSO the highest-overlap one. So: **don't
   symmetrize, don't concat — bibliographic coupling alone.** (This overturns the current `max(P,Pᵀ)`
   feature on both counts: internal→full and symmetrize→out-only.)
3. **The citation-fusion dead-end is REOPENED on the right feature.** The CCA/discard-clash 4/4 losses were
   measured against a near-identity noise matrix. The genuine residual still holds (coupling explains only
   ~15% of keyword variance — citations see method/community lineage, keywords see topic; partial, not
   redundant), so don't expect a knockout — but a real test is now warranted: **(a) cluster on full-coupling
   ALONE** (collaborative view, no embedder — the "use one or the other" elegance; rank is ample) and **(b)
   re-run CCA(keyword, full-coupling)** retaining the now-real shared subspace; judge both vs embedding.
4. **Whittling citations:** participation 1272 (out-coupling) means energy spreads over ~1k directions;
   the top few are the megahubs (foundational works everyone cites) = the citation analog of
   language/paratext. Whittling *the citation view on its own terms* (remove top-k hubs) is the principled
   move — NOT using keywords to whittle citations (or vice versa), which fails for the partial-overlap
   reason. Each view denoised by its own nuisance axis.

Net: the earlier dead-end's *diagnosis was right for the feature it measured*, but the feature was wrong.
The corrected reopening is the live follow-up; the validated default (embedding/CCA clustering on the
concept) is unaffected meanwhile.

### RESULT (2026-06-03): the FUSION wins embedder-free; citation-ALONE is coverage-capped

Ran both follow-ups (`cluster_concept_cca --citation-source full_coupling`, judge ×5 vs embedding):

**(b) CCA(keyword, full-coupling) — WINS 3/4 vs embedding** (interestingness/comprehensiveness/specificity;
ties progressive), **embedder-free**. Canonical ρ=[0.61,0.59,0.56,…], **k=21** retained (vs the internal
feature's ρ1=0.229/k=6) — a real, large shared subspace. genericness 8.94, n_tiny=3, ~2 paratext clusters,
clean modern curriculum. On jurafsky, **keyword TF-IDF + bibliographic coupling together fully substitute
for the neural embedder** (matches CCA-embedding's 3/4 jurafsky win). Direct payoff on "do we need the
embeddings at all?" — here, no.

**(a) citation-ONLY — LOSES 3/4, and is COVERAGE-CAPPED at 66.8%.** Two hard problems:
1. **3278/9886 works have ZERO in-field out-citations** (no collaborative signal) → dropped. Many are
   books/chapters (sparse/no parsed refs in OpenAlex) or works citing only outside field 17. **Matrix
   completion cannot rescue an all-empty row** (nothing to extrapolate from *within* the citation graph;
   you'd need side info = content, defeating "citation-only"). So the collaborative view is fundamentally
   capped ~⅔ on this corpus.
2. **Community ≠ topic**: redundant clusters — RNN/GRU across T1/T6/T17, backprop across T8/T11,
   classical-ML across T5/T16 — citation cliques fragment one topic and merge co-citing unrelated topics.

**Verdict on "use one OR the other" (the elegance):** refuted *for citation-alone* (coverage cap +
community-fragmentation). The win is the **FUSION** — keywords supply coverage (place the ⅓ uncited works)
and regularize citation-communities into topics; coupling supplies embedder-free language-robustness. Two
cheap text-derived views replace the neural embedder; neither alone suffices.

**Gate on completion / self-link (the design discussion):** the experiment answered it. WINNER (b) clusters
are clean (n_tiny=3) → completion/τ-regularization is a **no-op** for it. LOSER (a) is capped by *empty*
rows, which completion can't fill (only sparse-but-nonempty rows) → **insufficient**. So the τ-diffusion
ladder (Katz/PPR with a global τ≈d̄, the minimal completion = a self-link) is neither needed for the winning
path nor able to salvage the losing one *here*. It would only matter on a corpus where the collaborative
view is sparse-but-not-empty. Self-link in the coupling feature itself stays at 0 (stripped — a self-loop
spuriously couples a work with its own citers).

**Multi-book judge (2026-06-03): CCA-coupling is TOPIC-DEPENDENT, generally ≤ CCA-embedding — NOT a
universal GPU-free default.** gemma4 ×5, CCA(keyword, full_coupling) vs *both* embedding and CCA-embedding:
- **jurafsky** (multilingual NLP, ~8 refs/work, k=21): coupling **WINS 3/4 vs embedding**.
- **color_image** (N=24, 6.3 refs/work, k=23): coupling **TIES embedding** (loses only specificity); but
  CCA-**embedding > coupling** (wins interestingness + specificity).
- **mcs** (N=10, 11.7 refs/work, k=54): coupling **LOSES 4/4 vs embedding**; CCA-embedding > coupling (3/4).

So the GIST embedder adds **real value on the English-dominant niche books**; keyword+citation *matches* it
only on the citation-rich/multilingual case (jurafsky), where language-fragmentation hurts the keyword view
and the embedder's robustness is partly redundant with the citation agreement-subspace. **Verdict:**
CCA-coupling is a viable embedder-free option *for multilingual/citation-heavy topics*, NOT a universal
replacement — the embedder isn't redundant. embedding/CCA-embedding stays the default. This fits the
unified-views picture (keyword TF-IDF, bibliographic coupling, and the neural embedder are three
co-occurrence views of one latent topic structure — *which two suffice is topic-dependent*; see the
"using a word ≈ citing it" note). Code: `cluster_concept_cca --method {cca,citation_only} --citation-source
full_coupling`. (Aside: mcs kept k=54 canonical dirs for 10 chapters — elbow over-retention; a possible
knob, but doesn't flip the verdict.)

**The "using a word ≈ citing it" unification (worth a backlog of its own):** both views are work×vocabulary
incidence matrices (a work is a bag of terms OR a bag of references); TF-IDF≡megahub-down-weighting,
LSA-topic≡citation-community, word2vec≡node2vec, Firth's "company a word keeps"≡bibliographic coupling. A
citation is a *proper-noun word* (ultra-high-IDF, ultra-sparse) — which predicts coupling's specific-but-
sparse profile (high genericness, 67% coverage cap) directly. Formal home: a joint latent-factor model
emitting both words and citations (Relational Topic Model / Link-LDA; Chang & Blei 2009); ridge-CCA is its
linear-Gaussian spectral shadow. **Cheap direct test:** factor a single `[X ‖ G_out]` incidence matrix
(terms ⊕ cited-works) — "a reference is just another word" — vs CCA-agreement vs the generative model.

**RESULT (2026-06-03): reference-as-word works, but CCA's machinery IS load-bearing.** `--method
concat_vocab --concat-alpha 0.5` (each block L2-normed per row, energy-balanced, single LSA) on jurafsky:
**100% coverage** (terms place the uncited ⅓ — no zero rows), genericness 8.85. Judge ×5: **beats embedding
2/4** (comprehensiveness, specificity), ties the other 2 — so the isomorphism is *practically* valid; a
reference is usable as just another word, and a single SVD gives a competitive GPU-free curriculum. **BUT
CCA-coupling beats concat_vocab 3/4** → the two-view whitening + agreement-selection is NOT redundant.
The "why" is visible in the output: concat_vocab threw a **3198-work mega-cluster** (⅓ of works) + a few
garbled clusters = **term-domination** — terms have ~1M columns vs refs' ~22k, so even at equal per-row
energy the lexical block tilts the top SVD components and injects language noise (the foreign/garbled
clusters survive). CCA fixes both *for free*: whitening auto-balances the views scale-/rank-invariantly
(no α to tune), and agreement-selection collapses the view-specific language directions. **Net on the
concat↔CCA↔generative spectrum: the simple concat is a real, working fusion, but CCA is the sweet spot —
parameter-free and denoising.** (An α-sweep — lower α = more ref weight — might narrow the gap, but CCA
wins without a knob, so it's not worth a default.) Code: `cluster_concept_cca --method concat_vocab`.

## Status / working notes

- **2026-06-03:** idea captured. Confirmed framing: the concept fixed point is a *soft weight vector* that
  replaces the embedding ball at the target-works/personalization interface; all work types downstream;
  book-deficit-lift as the spin-off gap detector. Next: external-seed adaptation of `gap_keyword_iteration`.
- **2026-06-03:** BUILT (`gap_keyword_concept.py`) + the cached full-field index (`field_term_index.py`,
  `@MEMORY.cache`: field-17 6.84M works × 1M terms, 840M nnz, ~19 min one-time, instant reuse).
  - **Whole-ToC single seed COLLAPSES** to the tightest sub-basin (Jurafsky → all speech/ASR/perplexity) —
    the one-basin lesson again. Fix: **per-chapter union** (`--per-chapter`).
  - **`--backend nearest` (embedding-nearest-20k pool) basin-FUNNELS**: ~half of Jurafsky's chapters
    collapsed into one `pos_tagging` attractor (Embeddings, IR, RNNs, Phonetics, IE, SRL, Sentiment all
    landed there) because the pool is POS-heavy.
  - **`--backend full` + `--per-chapter --iters 3` WORKS BEAUTIFULLY.** Each of Jurafsky's 25 chapters
    recovered its *exact* distinct basin: Embeddings→word2vec/glove/fasttext, Alignment→rlhf/dpo,
    Transformers→self-attention, MT→nmt, Phonetics→mfcc, SRL→propbank, Discourse→pdtb, Dialogue→
    task-oriented, IR→rag, RNNs→lstm/gru, … (only "Introduction" is generic/paratext, as expected).
    The full field as candidates removed the POS funnel. **Validates keyword-concept seeding end-to-end
    on a popular book that had gap_size 0.**
  - **Spin-off signal fired:** the union concept's `book_lift = 0.49` (book-DEFICIT) — even mainstream NLP
    is proportionally preprint/article-heavy.
  - Focused the target to the **∪ of per-chapter basins** (9,886 works, not the 274k floor-support) and
    emit a `concept_gap.json` so `book_curriculum.py` consumes it as `target_work_ids`.
- **2026-06-03 — END-TO-END on Jurafsky (the answer to "does it work on a popular textbook"): YES.**
  `book_curriculum --gap-json concept_gap.json --semantic-rerank --dynamic-topk-floor 0.05
  --desired-n-topics 25` (path note: output landed at the doubled `artifacts/curriculum/artifacts/
  curriculum/jurafsky_trackA` — book_curriculum prepends `artifacts/curriculum/`). The **unsupervised**
  clustering (no ToC peeking) recovered **~20/25 of Jurafsky's chapters** as distinct topics —
  Phonetics(mfcc), N-gram LMs, Embeddings(word2vec), Neural Nets(backprop), MT(nmt), RNNs(lstm/gru),
  Dependency Parsing, Constituency Parsing(CFG), NER, SRL(propbank), Info-Extraction, Entity-Linking,
  Discourse(pdtb), Dialogue, Sentiment, Words/Spelling, TTS, ASR, Introduction(paratext).
  - **The misses are diagnostic: the RECENT frontier chapters** (Transformers, Masked-LM/BERT, LLMs,
    Post-training/RLHF, IR/RAG) are under-represented. The keyword-concept FOUND them all (basins had
    self-attention/bert/plms/rlhf-dpo/rag), but the builder's PageRank-centrality + product-of-experts
    selection DROPPED them. Quantified: concept = 55.6% works ≥2020; after selection = 36.0%. **The
    recency de-frontiering (the known weakness) is localized to the SELECTION stage**, not the concept.
  - Minor: mild redundancy (sentiment/NMT/speech each split across 2–3 topics).
  - **Upshot:** keyword-concept seeding works on a popular textbook; completing a faithful Jurafsky
    reconstruction now reduces to **fixing recency at the selection stage** (recency-aware re-rank, or
    seed-weighted personalization that preserves the concept's recency). Also TODO: skip the 6.8M-field
    background plot for full-field-concept runs (it's the slow tail); fix the doubled output-dir path.
- **2026-06-03 — RECENCY FIXED by decoupling selection from the citation walk.** Threaded the concept
  soft weights into the PageRank personalization (`cfg.target_work_weights` → non-uniform teleport),
  added `cfg.decouple_selection` (concept personalization IS the selection mass; citation PR only orders),
  and `--pagerank-alpha` / `--no-plot` on `book_curriculum`. `concept_gap.json` now carries
  `gap_article_weights`. Jurafsky, no-rerank, frac≥2020 of selected works:

  | variant | mean year | frac≥2020 | frontier chapters? |
  |---|---|---|---|
  | concept (target) | 2017.8 | 55.6% | — |
  | original (uniform + rerank) | 2015.8 | 36.0% | ✗ missing |
  | weighted personalization (α=.85) | **2011.5** | 50.5% | partial (bimodal) |
  | **decoupled (concept selects)** | **2018.9** | **63.6%** | ✓ Transformers, BERT, LLMs, RLHF |

  **Decoupling recovers recency AND the frontier chapters** (T9=rlhf/dpo/alignment, T12=transformer/
  self-attention, T2=bert, T15=plms/fine-tuning). **Weighted-α=.85 is bimodal** (50.5% ≥2020 but mean
  *2011.5*) — concept teleport keeps recent works but α-diffusion still drags in old citation hubs,
  proving the citation walk is the de-frontiering culprit (weighting alone insufficient; tame α or
  bypass it). The α-sweep (`--pagerank-alpha`) is the diagnostic dial between the two. **Verdict: use
  `--decouple-selection` for concept-seeded curricula** (citations order, concept selects).
  - Residual (separate from recency): embedding clustering over-allocates to dense regions (6 speech /
    3 paratext topics on Jurafsky) — a balance issue, not the frontier one.
