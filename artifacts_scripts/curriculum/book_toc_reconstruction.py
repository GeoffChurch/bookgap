"""Track B + reconstruction eval for the real-book ToC experiment.

Track B: turn a real book's ToC into "our format" — embed each chapter
(title+description) into the corpus embedding space and assign each of the book's
own-gap articles to its nearest chapter. This is the ground-truth partition.

Eval: compare our curriculum (Track A: book_curriculum.py output) against the
ToC-induced partition, over the shared gap articles:
  - clustering agreement: Adjusted Rand Index + Normalized Mutual Info
  - per-chapter purity: best-overlap Track-A cluster per ToC chapter, matched frac
  - ordering: Kendall-tau between ToC chapter order and our frontier order on
    matched (chapter -> cluster) pairs

Raw GIST-small embedding first (domain-adaptation step 1); a --mean-center flag
applies the cheap per-genre offset removal (step 2).

Example::

    .conda/bin/python -m artifacts_scripts.curriculum.book_toc_reconstruction \\
        --gap-json artifacts/book_gaps/W2898513995_cyberbullying_ai.json \\
        --toc-json artifacts/book_gaps/W2898513995_toc.json \\
        --curriculum-dir artifacts/curriculum/book_curriculum_W2898513995_cyberbullying \\
        --out artifacts/book_gaps/W2898513995_reconstruction.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.stats import kendalltau
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from artifacts_scripts.curriculum.config import CurriculumConfig
from bookgap.subcloud import SubCloud
from bookgap.util import Filter, LOGGER, setup_logging


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


_STOP = set((
    "the of for and or to in on with using based via from into over under a an is are be "
    "this that these those we our their it its as at by such new approach approaches method "
    "methods system systems model models use survey review study analysis toward towards "
    "via case study via data driven").split())


def _materialize_track_b(out_dir: Path, chapters: list, g_titles: list,
                         sims: np.ndarray, b_label: np.ndarray, book_title: str,
                         n_works: int = 5, n_kw: int = 10) -> None:
    """Write Track B (the book's ToC, populated with the assigned gap articles) as a
    renderable curriculum run-dir (summary.md + pipeline.log), in the SAME format
    eval_book_judge.load_book_outline parses -- so the pairwise judge renders it
    blind/symmetric vs our Track A curriculum (labels come from keywords, not the
    book's real chapter titles).
    """
    import math
    import re
    from collections import Counter

    def toks(s: str) -> list:
        ws = [w for w in re.findall(r"[a-z][a-z\-]{2,}", s.lower()) if w not in _STOP]
        return ws + [f"{ws[i]} {ws[i + 1]}" for i in range(len(ws) - 1)]  # unigrams + bigrams

    n_chap = len(chapters)
    chap_tf = [Counter() for _ in range(n_chap)]
    for j, t in enumerate(g_titles):
        chap_tf[int(b_label[j])].update(toks(t))
    df: Counter = Counter()
    for ci in range(n_chap):
        for w in chap_tf[ci]:
            df[w] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Curriculum Summary: {book_title}", ""]
    for ci in range(n_chap):
        assigned = sorted((j for j in range(len(g_titles)) if int(b_label[j]) == ci),
                          key=lambda j: -float(sims[j, ci]))               # nearest first
        idf = {w: math.log((n_chap + 1) / (df[w] + 1)) + 1.0 for w in chap_tf[ci]}
        kws = sorted(chap_tf[ci], key=lambda w: -chap_tf[ci][w] * idf[w])[:n_kw]
        lines += [f"## Topic {ci + 1}", "", "### Top Keywords", "",
                  "| Keyword | Score |", "|---------|-------|"]
        for w in kws:
            lines.append(f"| {w} | {chap_tf[ci][w]} |")
        lines += ["", "### Top Works", ""]
        for k, j in enumerate(assigned[:n_works], 1):
            lines.append(f"#### {k}. {g_titles[j]}")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))
    (out_dir / "pipeline.log").write_text(
        f"Topic T0000 ({book_title}) gap 0: materialized ToC curriculum (Track B)\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-json", required=True, type=Path)
    ap.add_argument("--toc-json", required=True, type=Path)
    ap.add_argument("--curriculum-dir", required=True, type=Path)
    ap.add_argument("--mean-center", action="store_true",
                    help="domain-adaptation step 2: subtract per-genre means before cosine")
    ap.add_argument("--materialize-dir", type=Path, default=None,
                    help="also write Track B as a renderable curriculum run-dir (summary.md) "
                         "for the pairwise LLM eval (eval_pairwise_judge).")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    # Attach a per-record-flushing FileHandler (+ console) so LOGGER.info isn't
    # dropped by the handler-less "bookgap" logger (see book_own_gap).
    setup_logging(log_name="book_toc_reconstruction", log_level="INFO",
                  include_time=False, log_path=args.out.with_suffix(".log"))

    gap = json.loads(args.gap_json.read_text())
    bk = gap.get("book", gap)
    gap_ids = bk["gap_article_ids"]
    subfield = gap.get("scope_subfield")
    field = gap.get("scope_field")
    toc = json.loads(args.toc_json.read_text())
    chapters = toc["chapters"]

    # --- load scope cloud + article subcloud (same scope the gap was computed at)
    cfg = OmegaConf.structured(CurriculumConfig())
    if subfield:
        cfg.experiment.cloud.filter_pair.pre.subfield = str(subfield)
    if field:
        cfg.experiment.cloud.filter_pair.pre.field = str(field)
    cloud = instantiate(cfg.experiment.cloud)
    articles = SubCloud(parent=cloud, post_filter=Filter(work_types=("article",)))
    art_ids = articles.work_ids()
    id2row = {a: i for i, a in enumerate(art_ids)}
    art_emb = np.asarray(articles.embeddings(), dtype=np.float32)
    titles = articles.work_titles()

    gap_rows = [(a, id2row[a]) for a in gap_ids if a in id2row]
    g_ids = [a for a, _ in gap_rows]
    g_emb = _l2(art_emb[[r for _, r in gap_rows]])
    LOGGER.info("gap articles resolved in scope: %d/%d", len(g_ids), len(gap_ids))

    # --- Track B: embed chapters in the same space, assign each gap article
    embedder = instantiate(cfg.experiment.cloud.embedder)
    with embedder as emb:
        chap_emb = np.asarray(emb.embed_texts([(c["title"], c.get("description", "")) for c in chapters]),
                              dtype=np.float32)
    if args.mean_center:
        g_emb = _l2(g_emb - g_emb.mean(0))
        chap_emb = chap_emb - chap_emb.mean(0)
    chap_emb = _l2(chap_emb)
    sims = g_emb @ chap_emb.T                       # (n_gap, n_chap)
    b_label = sims.argmax(1)                         # ToC chapter per gap article
    chap_titles = [c["title"] for c in chapters]
    b_sizes = np.bincount(b_label, minlength=len(chapters))
    LOGGER.info("Track B chapter assignment sizes: %s",
                {chap_titles[i][:28]: int(b_sizes[i]) for i in range(len(chapters))})

    if args.materialize_dir:
        g_titles = [titles[r] for (_, r) in gap_rows]
        _materialize_track_b(args.materialize_dir, chapters, g_titles, sims, b_label,
                             toc.get("title", "Book"))
        LOGGER.info("materialized Track B run-dir -> %s", args.materialize_dir)

    # --- Track A: our curriculum clusters + frontier ordering
    twi = json.loads((args.curriculum_dir / "topic_work_ids.json").read_text())
    # topic_work_ids is keyed "0".."N-1" in frontier (chapter) order — that IS the
    # ordering (metadata.frontier_topic_ids lists the same topics' internal leaf ids
    # in the same order), so the order position of cluster key c is just int(c).
    order_pos = {c: int(c) for c in twi.keys()}
    work2cluster = {w: c for c, ws in twi.items() for w in ws}

    a_label_map = {a: work2cluster.get(a) for a in g_ids}
    shared = [(a, i) for i, a in enumerate(g_ids) if a_label_map[a] is not None]
    LOGGER.info("gap articles also in our curriculum clusters: %d/%d", len(shared), len(g_ids))
    if len(shared) < 5:
        LOGGER.warning("too few shared articles for a meaningful comparison")

    a_lab = [a_label_map[a] for a, _ in shared]
    b_lab = [int(b_label[i]) for _, i in shared]

    ari = float(adjusted_rand_score(b_lab, a_lab))
    nmi = float(normalized_mutual_info_score(b_lab, a_lab))

    # per-chapter purity: best-overlap A-cluster per ToC chapter
    clusters = sorted(set(a_lab))
    matches = []
    matched = 0
    for ci in range(len(chapters)):
        idxs = [j for j, bl in enumerate(b_lab) if bl == ci]
        if not idxs:
            continue
        counts = {}
        for j in idxs:
            counts[a_lab[j]] = counts.get(a_lab[j], 0) + 1
        best_cl, best_n = max(counts.items(), key=lambda kv: kv[1])
        matched += best_n
        matches.append({"chapter_n": chapters[ci]["n"], "chapter": chap_titles[ci],
                        "n_articles": len(idxs), "best_cluster": best_cl,
                        "overlap": best_n, "cluster_order_pos": order_pos.get(best_cl)})
    purity = matched / max(1, len(shared))

    # ordering: ToC chapter order (n) vs matched cluster frontier position
    pairs = [(m["chapter_n"], m["cluster_order_pos"]) for m in matches if m["cluster_order_pos"] is not None]
    tau = float("nan")
    if len(pairs) >= 3:
        tau, _ = kendalltau([p[0] for p in pairs], [p[1] for p in pairs])

    result = {
        "book_id": bk.get("book_id"), "scope_subfield": subfield, "mean_center": args.mean_center,
        "n_gap_articles": len(gap_ids), "n_resolved": len(g_ids), "n_shared_with_curriculum": len(shared),
        "n_toc_chapters": len(chapters), "n_our_clusters": len(clusters),
        "adjusted_rand_index": ari, "normalized_mutual_info": nmi,
        "chapter_purity": purity, "ordering_kendall_tau": tau,
        "track_b_chapter_sizes": {chap_titles[i]: int(b_sizes[i]) for i in range(len(chapters))},
        "chapter_matches": matches,
    }
    LOGGER.info("RECONSTRUCTION: ARI=%.3f NMI=%.3f purity=%.3f order_tau=%.3f (chapters=%d clusters=%d shared=%d)",
                ari, nmi, purity, tau, len(chapters), len(clusters), len(shared))
    for m in matches:
        LOGGER.info("  ch%-2d %-46s -> cluster %s (%d/%d)", m["chapter_n"], m["chapter"][:46],
                    m["best_cluster"], m["overlap"], m["n_articles"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    LOGGER.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
