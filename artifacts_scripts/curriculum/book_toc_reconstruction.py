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
from bookgap.util import Filter, LOGGER


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-json", required=True, type=Path)
    ap.add_argument("--toc-json", required=True, type=Path)
    ap.add_argument("--curriculum-dir", required=True, type=Path)
    ap.add_argument("--mean-center", action="store_true",
                    help="domain-adaptation step 2: subtract per-genre means before cosine")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    gap = json.loads(args.gap_json.read_text())
    bk = gap.get("book", gap)
    gap_ids = bk["gap_article_ids"]
    subfield = gap.get("scope_subfield")
    toc = json.loads(args.toc_json.read_text())
    chapters = toc["chapters"]

    # --- load scope cloud + article subcloud (same scope the gap was computed at)
    cfg = OmegaConf.structured(CurriculumConfig())
    if subfield:
        cfg.experiment.cloud.filter_pair.pre.subfield = str(subfield)
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
