"""Do a book's CHAPTERS fill gaps, even when the whole book's own-gap is ~0?

Canonical textbooks sit in book-dense regions -> own-gap ≈ 0 (book_own_gap.py).
But each chapter is far more specific, so chapter-level gaps may be non-empty.
We treat each ToC chapter (title+description) as a query point (embedded with the
corpus GIST embedder, like book_own_gap's --query-text path) and score the gap it
sits in under three definitions:

  raw      : # articles within radius_fraction x dist-to-nearest-book (raw k-ball).
  kde      : KDE-on-KDE residual mass of that ball (gap_size_metrics.kde_ratio) —
             discounts the count by how book-claimed the neighborhood is.
  maxratio : max over the k nearest books of (#articles closer than book j)/j —
             the greatest article-to-book ratio over all radii out to the k-th
             nearest book (a multi-scale density-ratio; reports the scale j*).

Loads the cloud ONCE and scores every chapter of every --toc passed.

    PYTHONPATH=. .conda/bin/python -m artifacts_scripts.curriculum.chapter_gap \
        --toc artifacts/book_gaps/jurafsky_toc.json artifacts/book_gaps/koehn_toc.json \
        --subfield 1702 --out artifacts/book_gaps/chapter_gaps.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from artifacts_scripts.curriculum.config import CurriculumConfig
from bookgap.gap_size_metrics import kde_ratio
from bookgap.subcloud import SubCloud
from bookgap.util import Filter, LOGGER, setup_logging

RF = 0.75
KMAX = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toc", nargs="+", required=True)
    ap.add_argument("--subfield", default=None)
    ap.add_argument("--field", default=None)
    ap.add_argument("--radius-fraction", type=float, default=RF)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    setup_logging(log_name="chapter_gap", log_level="INFO", include_time=False,
                  log_path=args.out.with_suffix(".log"))

    cfg = OmegaConf.structured(CurriculumConfig())
    if args.subfield:
        cfg.experiment.cloud.filter_pair.pre.subfield = str(args.subfield)
    if args.field:
        cfg.experiment.cloud.filter_pair.pre.field = str(args.field)
    cloud = instantiate(cfg.experiment.cloud)
    LOGGER.info("Cloud: %s", cloud.name())
    books = SubCloud(parent=cloud, post_filter=Filter(work_types=("book",)))
    articles = SubCloud(parent=cloud, post_filter=Filter(work_types=("article",)))
    b_emb = np.asarray(books.embeddings(), dtype=np.float32)
    a_emb = np.asarray(articles.embeddings(), dtype=np.float32)
    b_titles = books.work_titles()
    LOGGER.info("scope: %d books, %d articles", len(b_emb), len(a_emb))
    b2 = np.einsum("ij,ij->i", b_emb, b_emb)
    a2 = np.einsum("ij,ij->i", a_emb, a_emb)

    embedder = instantiate(cfg.experiment.cloud.embedder)
    out = {}
    for toc_path in args.toc:
        toc = json.loads(Path(toc_path).read_text())
        chs = toc["chapters"]
        texts = [("", f"{c.get('title','')}. {c.get('description','')}") for c in chs]
        with embedder as e:
            qv = np.asarray(e.embed_texts(texts), dtype=np.float32)
        qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)

        rows = []
        print(f"\n=== {toc['title']} ({len(chs)} chapters) ===")
        print(f"{'ch':>3} {'raw':>5} {'kde':>6} {'maxratio':>9} {'j*':>3}  nearest book / chapter")
        for ci, c in enumerate(chs):
            q = qv[ci]
            d2b = float(q @ q) + b2 - 2.0 * (b_emb @ q)
            order = np.argsort(d2b)
            d_nn = float(d2b[order[0]])
            radius = args.radius_fraction * d_nn
            d2a = float(q @ q) + a2 - 2.0 * (a_emb @ q)
            members = np.where(d2a <= radius)[0]
            raw = int(members.size)
            kde = kde_ratio(members.astype(np.int64), a_emb, b_emb)
            best_ratio, best_j, best_a = 0.0, 0, 0
            for j in range(1, min(KMAX, len(order)) + 1):
                r_j = float(d2b[order[j - 1]])
                a_j = int((d2a <= r_j).sum())
                if a_j / j > best_ratio:
                    best_ratio, best_j, best_a = a_j / j, j, a_j
            nb = str(b_titles[int(order[0])])[:40]
            print(f"{c.get('n'):>3} {raw:>5} {kde:>6.1f} {best_ratio:>9.1f} {best_j:>3}  "
                  f"{nb}  | {str(c.get('title',''))[:24]}")
            rows.append({"n": c.get("n"), "title": c.get("title"), "raw": raw,
                         "kde": round(kde, 2), "maxratio": round(best_ratio, 2),
                         "j_star": best_j, "nearest_book": nb})
        raws = [r["raw"] for r in rows]
        print(f"  book summary: chapters_with_gap>0 = {sum(x > 0 for x in raws)}/{len(rows)}; "
              f"raw max={max(raws)} median={int(np.median(raws))}; "
              f"kde max={max(r['kde'] for r in rows):.0f}")
        out[toc["title"]] = rows

    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
