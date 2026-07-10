"""Corpus-free firm-up of the ordering/good-edges genre finding, across all harvested ToCs.

Tests whether a book's chapters are ordered by LOCAL topical adjacency: is the mean cosine
similarity of CONSECUTIVE chapters higher than under random chapter orderings? (Self-contained
— embeds the ToC's own chapters, no field corpus needed.) Reports a per-book percentile and
splits by genre (single-author textbook vs edited/niche volume) to firm the n≈4 claim that
textbooks order by good edges and edited volumes don't.

    CUDA_VISIBLE_DEVICES="" uv run python -m artifacts_scripts.curriculum.ordering_genre_firmup
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

# Edited/niche research volumes (the original contrast set); everything else = single-author textbook.
EDITED = {"W2898513995", "W4211017872", "W4235013513", "W4247813589"}
N_PERM = 4000
SEED = 0


def main() -> None:
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    rng = np.random.default_rng(SEED)
    rows = []
    for f in sorted(glob.glob("artifacts/book_gaps/*_toc.json")):
        slug = Path(f).stem.replace("_toc", "")
        toc = json.loads(Path(f).read_text())
        chs = toc.get("chapters") or []
        if len(chs) < 6:
            continue
        texts = [f"{c.get('title','')}. {c.get('description','')}" for c in chs]
        E = m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        S = E @ E.T
        n = len(chs)

        def mean_edge(order):
            return float(np.mean([S[order[k], order[k + 1]] for k in range(n - 1)]))

        actual = mean_edge(list(range(n)))
        rand = np.array([mean_edge(list(rng.permutation(n))) for _ in range(N_PERM)])
        pct = float((rand < actual).mean())
        genre = "edited" if toc.get("book_id", "").rsplit("/", 1)[-1] in EDITED or slug in EDITED \
            else "textbook"
        rows.append({"slug": slug, "n": n, "good_edge": round(actual, 3),
                     "rand_mean": round(float(rand.mean()), 3), "pctile": round(pct, 3),
                     "genre": genre})

    rows.sort(key=lambda r: -r["pctile"])
    print(f"{'book':24}{'n':>3}{'good-edge':>10}{'rand':>7}{'pctile':>8}  genre")
    for r in rows:
        print(f"{r['slug']:24}{r['n']:>3}{r['good_edge']:>10.3f}{r['rand_mean']:>7.3f}"
              f"{r['pctile']:>8.0%}  {r['genre']}")
    for g in ("textbook", "edited"):
        ps = [r["pctile"] for r in rows if r["genre"] == g]
        if ps:
            hi = sum(p >= 0.90 for p in ps)
            print(f"\n{g:9} n={len(ps)}  median pctile={np.median(ps):.0%}  "
                  f"mean={np.mean(ps):.0%}  ≥90th: {hi}/{len(ps)}")
    Path("artifacts/book_gaps/ordering_genre_firmup.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote artifacts/book_gaps/ordering_genre_firmup.json ({len(rows)} books)")


if __name__ == "__main__":
    main()
