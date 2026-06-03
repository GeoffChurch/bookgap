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
