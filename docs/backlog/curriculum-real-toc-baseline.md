# Real-book ToC reconstruction eval (book-first, whole-domain, book's-own-gap)

**The design.** Manually pick real OA books (any topic) that have recoverable ToCs. For
each book, compute **its own gap** the way the main pipeline already does — treat the book
as a mere article and find its **nearest non-self neighboring book**, k-ball around it —
at the **whole Physical-Sciences-domain** scope. Seed *our* curriculum from that gap-ball,
and score it against the book's **own ToC converted to our format** as ground truth. Gap
validity is free (a real book filled it), gap selection has no confound (the book defines
its own location), and the eval is a quantitative reconstruction score, not a 1–7 vibe.

This supersedes the earlier "start from our computed gaps" framing: we do **not** use the
gap detector's chosen gaps. The book is the unit; its gap is intrinsic to where it sits.

## Why book-first / whole-domain is the right call
- **Kills the gap-selection confound.** Any OA book is a clean test case. The book's own
  gap percentile becomes a free **difficulty axis** — sample books from deep-gap to
  crowded-region.
- **Whole-domain scope is honest.** The nearest-non-self book must be found among a
  comprehensive book set; a narrow (single-topic) scope makes a book look gap-filling only
  because its real neighbors outside the topic were excluded (boundary artifact). The
  curriculum pipeline already runs at `domain=3` Physical Sciences (40.9M works), so this
  needs **no new infrastructure**. Report the percentile at *both* scopes (domain-wide =
  absolute gap size; AI-subfield = interpretable) — recomputing percentile against a book
  subset is near-free.
- **Reuses confirmed machinery.** Verified the book's-gap mechanism in code:
  `gaps.py:97-101` scores each book as a query point (`is_intra_cloud=True`);
  `kball_algorithm.py:203` `effective_k = k+1 if is_intra_cloud` drops the self-match (the
  "non-self" part); `gap_statistics.py:41` percentiles against the per-book distribution;
  both `gap_size_percentile` and `proposal_gap_size_percentile`
  (`results_export.py:112,291`) use it.

## Thin new entry point: book → its-own-gap → seed
`seed_from_gap` currently loads in-gap article ids from a precomputed gap context. Swap
that front-end for the book's own gap-ball; everything downstream is unchanged.

```python
# domain-scope FilterPair -> books (landmark) + articles (ambient) subclouds
from bookgap.cloud_index import get_kball_coverage
book_idx   = books.index_of(book_openalex_id)            # the chosen OA book
one_book   = books.subset([book_idx])                    # 1-point query cloud
cov = get_kball_coverage(ambient=articles, landmark=books, query=one_book,
                         radius_fraction=RF, k=1, is_intra_cloud=True)  # self-book dropped
_dists, idxs = cov[0]                                    # articles inside the book's ball
seed_article_ids = [articles.works()[i]["id"] for i in idxs]
# -> build_curriculum_for_gap(ids=seed_article_ids, ...) with the book held out of landmark
```
No greedy deflation (that's for ranking many gaps); we want the single raw ball around the
book. `is_intra_cloud=True` gives the nearest *other* book → radius = `RF × that dist`.

## Pipeline
1. **Curate ToC books (any topic).** Search OA technical books with recoverable ToCs
   (GitHub LaTeX: D2L/MML/PML/OpenStax; DOAB/OAPEN; LectureBank), match to the corpus by
   DOI/title. The covering-books scan (`/tmp/toc_candidates.json`) is *one* source, not the
   selector — e.g. *Deep Learning on Graphs*, *Quantum Information Theory* (92%, T10020),
   *Interpretable ML* (Molnar, free online), *Introduction to Coding Theory* are all
   corpus-present textbooks.
2. **Two-stage to control cost.** A full curriculum is a ~104 GB, ~20-min run (one-at-a-
   time). So first cheaply compute the *gap-ball* (one `get_kball_coverage` call) for many
   candidate books, then run the full curriculum only for a chosen spread across the gap-
   percentile range.
3. **Track A (ours).** `build_curriculum_for_gap` on the book's gap-ball articles, book
   held out of the landmark set (so we reconstruct, not copy). Boundary/similarity variants
   apply.
4. **Track B (reference).** Parse the ToC; per chapter, embed title+desc into our space and
   retrieve nearest corpus articles + keywords → the book's ToC in our format. Domain-
   adaptation sequencing, cheapest first: raw GIST-small → per-genre mean-center/whiten →
   synthetic-abstract paraphrase (LLM rewrites chapter as abstract) → learned Procrustes/CCA
   (reuses [[curriculum-generalization-pedagogical-subspace]]).
5. **Eval — reconstruction quality (the payoff).** Track A vs Track B, quantitative:
   sub-topic recall (bipartite chapter↔cluster match by embedding sim / earth-mover);
   article recall (cousin of [[heldout-book-recovery-production-eval]],
   [[book-rediscovery-eval-candidates]]); ordering Kendall-τ on matched chapters. Secondary:
   LLM-judge pairwise (ours vs ToC-as-curriculum) — now the gold arm is a real book.

## Caveats
- **Genre mismatch** (chapter-title+intro vs title+abstract) applies to Track B — cheap
  domain-adaptation fixes before anything learned.
- **Leakage**: hold the book out of the landmark set for Track A. Its *own* gap computation
  self-excludes already.
- **ToC availability** gates the candidate set; report which (book, gap) pairs are in scope.

## Status / working notes (2026-06-01)
Entry point implemented: `artifacts_scripts/curriculum/book_own_gap.py` (querying the whole
book set with `is_intra_cloud=True` gives every book's own gap + the gap-size distribution
in one pass). Validated at **AI-subfield (1702)** scope, radius_fraction=0.75:
- **Popular books have tiny own-gaps.** *Deep Learning on Graphs* (W3160730739, the 81%
  coverer of the T11273 detected gap) → own-gap of only **14 articles**, all near-duplicate
  echoes (other GNN survey books / workshop proceedings). Its nearest non-self book is
  essentially a duplicate → tiny radius. Confirms the gap-vs-established tension: a great
  *coverer* of an article-gap can sit in a *book-dense* region with a near-zero own-gap.
- **Distribution is book-dense**: over 14,591 AI books, median own-gap=0, p90=3, p99=27,
  max=465. Only the top ~1% have a usable own-gap.
- **Top own-gaps split into (a) genuine niche textbooks with coherent gaps** (Federated ML
  on constrained devices 465, Cyberbullying Detection 341, Neuromorphic Photonics 280,
  Semantic Segmentation 141, Multitask Learning 134, Extreme Learning Machine 125) **and
  (b) debris/misclassification outliers** (a Tamil reference grammar → DOI-string garbage; a
  monastery-history book; Indonesian/Thai/Arabic student-thesis clusters).
- **Selection rules learned:** (1) require a *recoverable ToC* — it naturally selects real
  textbooks and drops debris; (2) prefer **large absolute own-gap**, not just high
  percentile (percentile is misleading when the median is 0); (3) **whole-domain scope**
  should shrink the misclassification false-gaps (photonics/crypto books misfiled into AI
  would find true neighbors). Ranked candidates: `artifacts/book_gaps/ai_subfield_ranked.json`.

**Full loop validated end-to-end** (AI scope) on *Automatic Cyberbullying Detection*
(W2898513995, own-gap 341 articles): `book_curriculum.py` (Track A) +
`book_toc_reconstruction.py` (Track B + eval). Results (341 gap → 219 shared with our
15-cluster curriculum; 8 ToC chapters):
- raw GIST-small: ARI=0.074, NMI=0.195, purity=0.30, order-τ=0.05.
- **+ mean-centering** (domain-adapt step 2): ARI=0.112, NMI=0.221, purity=0.36,
  **order-τ=0.34**. Validates the cheap-first sequencing — mean-centering removes the common
  "cyberbullying" component that made generic chapters (Background, Conclusion) absorb
  most articles, rebalancing the partition (Background 120→68, Android-Blocker 0→33) and
  letting our prereq order recover the chapter sequence.
- **Absolute reconstruction is modest** because this is a narrow *research monograph* (the
  authors' own niche methods: web-mining / brute-force / their Android app), not a broad
  field-structure textbook → method-oriented chapters don't match the literature's topic
  clusters. **Next: a broad textbook** (e.g. *Deep Learning on Graphs* would need a
  larger-own-gap niche book; *Quantum Information Theory* T10020 92% is a candidate) where
  ToC chapters reflect field structure. Eval machinery is ready; always apply mean-center.
- Code (uncommitted, exploratory): `book_own_gap.py`, `book_curriculum.py`,
  `book_toc_reconstruction.py`. ToC: `artifacts/book_gaps/W2898513995_toc.json`.

## Cross-refs
- [[curriculum-generalization-pedagogical-subspace]] — alignment machinery for Track B.
- [[curriculum-diffusion-and-quality-metrics]] — LectureBank source + intrinsic metrics.
- [[curriculum-ranking-input-levers]] — the structuring the reconstruction score isolates.
- Code: `bookgap/gaps.py`, `bookgap/llm/gap_algorithms/kball_algorithm.py`,
  `bookgap/gap_statistics.py`, `artifacts_scripts/curriculum/seed_from_gap.py`;
  candidate scan saved at `/tmp/toc_candidates.json`.
