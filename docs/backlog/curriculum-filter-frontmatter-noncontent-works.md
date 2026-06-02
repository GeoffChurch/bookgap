# Catalog-blurb abstracts (the "front-matter" debris — re-diagnosed)

**Correction (2026-06-02): this is NOT front-matter / non-content / paratext pollution.** Empirical
check of Color Image's debris chapter: the dominant work is `W1627054999` =
**"Two-Dimensional Signal and Image Processing"** (Jae Lim) — a *legitimate, on-topic* image-processing
textbook, `type=book`, `is_paratext=False`. We want it. Its stored OpenAlex "abstract" is a
**publisher catalog blurb**:

> "New to P-H Signal Processing Series (Alan Oppenheim, Series Ed) this text covers the principles
> and applications of multidimensional and digital signal processing. For Sr/grad level courses in
> image processing in EE departments."

Every junk keyword (`series, oppenheim, alan, departments, grad, courses, covers, text`) comes from
that one blurb. So `is_paratext`/type filtering would do nothing (the flag is correctly False), and
dropping the work would lose a relevant textbook.

**Real diagnosis — a composite of two known issues:**
1. **Degenerate cluster again.** This chapter is *one dominant work* (Lim). The blurb tokens score
   high only because Lim sits alone — in a normal 30-work image-processing chapter those tokens
   appear in 1/30 abstracts, score ~0, and never surface. dynamic-k can't help because in an
   isolated chapter the blurb *is* the high-mass content.
2. **Blurb abstract instead of a content abstract** — and this pollutes the **embedding** (Lim is
   placed in the space partly by marketing copy), not just the keywords. That hurts gap detection /
   selection / the seed KDE, which no downstream keyword fix recovers.

**Fix, in order:**
1. **Structural (real fix): [[curriculum-fuzzy-soft-topic-membership]].** Don't let a single work be
   a chapter; once Lim merges with the other image-processing works the blurb tokens dilute to ~0
   and vanish. Fixes BOTH Neuromorphic's and Color Image's structural debris at the root, no
   per-book heuristics. Highest leverage.
2. **Abstract hygiene (upstream, complementary, matters more than keywords):** detect catalog-blurb
   abstracts by telltale phrasing (`New to … Series`, `Series Ed`, `For … courses in … departments`,
   `this text covers`) and strip the boilerplate sentence or flag the abstract low-quality. Worth it
   because the blurb corrupts the embedding, upstream of everything.
3. **Keyword boilerplate stoplist (band-aid):** add publisher-metadata terms (`series`, editor names,
   `departments`, `grad`, `courses`) to the stopword filter. Whack-a-mole; doesn't touch embeddings.

**Salvaged from the old (wrong) framing:** a true paratext/`is_paratext` filter is still mild hygiene
worth having generally, but it is NOT what fixes this case. Eval via the ToC-reconstruction panel —
Color Image specificity is the metric to watch (the only panel book still losing it).
