# Deflation demand-engineering for a no-branch universal winner — refuted (metric artifact)

## The promising idea

`fl_bookdemand` (facility-location over book-neighborhood demand) won
book-rich/modern topics decisively but collapsed on book-sparse ones;
`submodular_fl` (manifold coverage) was the opposite. The seductive
premise: engineer a *single* demand/ranking objective whose book-vs-
manifold balance **emerges from the data**, so one no-branch, no-tuned-
constant method wins every regime. A long sequence of increasingly
clever variants pursued this.

## What was tried (all refuted)

- **A-keyed `fl_bookdemand_eb`** — books ∪ a manifold background
  coreset; shrinkage emergent from the customer-count ratio `M/(M+
  n_books)`, `M = clip(A//k_book,64,4000)`. *Refuted:* `M` keyed to
  *article* density ⇒ background dominates 6:1–38:1 on every topic ⇒
  ≈`submodular_fl` everywhere; forfeits the dense wins.
- **n_books-keyed constant-`s` shrinkage** — `M = n_books·s/(1−s)`
  pins shrinkage to a constant `s`, regime-independent. *Refuted:* a
  constant pins the *same* operating point everywhere, so it cannot be
  optimal in both regimes. The `s30/s50/s70` sweep was a frontier map,
  not winners (saturation tweaks of the same idea — no separate
  entry warranted). A hardcoded `s=f(book-density)` schedule was
  rejected pre-implementation: its endpoints (≈.2/≈.5) rested on a
  single sparse topic and were calibrated to the held-out metric
  (circular / overfit).
- **Intrinsic residual demand keyed to article count**
  (`M_bg=|O|//k_book`) — *Refuted in code review by algebra:* for
  clustered (i.e. all real) books `|O|≈A`, so it reduces to
  `≈A/(k_book·n_books)` — exactly the refuted A-keyed EB. (The
  *per-book*-normalized form `M_bg=|O|//max(k_book·n_books,k_book)` is
  the one that survived → that is the validated winner, see
  `docs/gap_deflation_results.md`; only the article-keyed form is the
  dead end here.)
- **`fl_union_bd_sub`** (union of fl_bookdemand + submodular_fl,
  dedup downstream) and **`fl_gen_rank`** (submodular generates the
  pool, book-demand reranks). *Refuted:* `fl_gen_rank` came out
  byte-identical to `submodular_fl`; `fl_union_bd_sub`'s apparent
  book-sparse "SOTA" (.625) evaporated under the corrected metric.
- **Ranking-stage book-aware selection** (`--rank effsize_x_bd`,
  `bookdemand`) — rank the robust submodular pool by
  `effective_size·(1+bd)`. *Refuted:* genuine recall@20 ≈ .0–.034 —
  book-aware *ranking of a book-blind pool* cannot manufacture
  book-shaped top gaps the pool's geometry never contained.

## The unifying diagnosis (the instructive part)

Every variant above was evaluated against `heldout_book_recall_at_20`,
which — through a misnomer in `run_arm` — was computed over the **full
deduped survivor set, not the top-20**. It therefore measured *set
coverage* and **rewarded emitting many gaps**, rank-insensitively. That
single defect manufactured the entire problem these variants were
solving:

- It made `submodular_fl`/`union` (which emit ~800–1100 gaps) look
  competitive (.38–.625) when their genuine recall@20 is ≈0.0.
- It made `fl_gen_rank` ≡ `submodular_fl` (same survivor *set*, and the
  metric never saw rank) — which was mis-read as "ranking is the
  binding constraint."
- It produced a phantom "fundamental book-vs-manifold tension", a
  regime gate, and a multi-week best-of-both search.

Correcting the metric to genuine `recall@20` (`ranked[:20]`) collapsed
all of it: coverage methods → ≈0.0, **no tension**, and the simplest
intrinsic book-anchored deflation (`fl_bookdemand_resid`,
per-book-normalized) wins outright.

## Don't re-run

Do not revisit demand-mixture / shrinkage-schedule / union / gen-rank /
ranking-stage engineering "to resolve the book-vs-manifold tension":
**there is no tension under a correct top-k recall metric.** Any future
held-out-book conclusion must use genuine recall@k (a real `ranked[:k]`
slice), and the metric's *implementation* must be read before trusting
any comparative sweep — a plausible column name (`..._at_20`) was not
proof of a top-20 computation. Validated outcome + full lesson:
`docs/gap_deflation_results.md`.
