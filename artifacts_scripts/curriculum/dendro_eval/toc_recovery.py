"""ToC-anchored dendrogram-recovery eval — the DECISIVE Track-2 experiment.

Track 1 (``intrinsic_dendro_sweep.py``) found that scipy **average-linkage**
preserves embedding distances far better (cophenetic ~0.65) than the production
default ``weighted_bisecting_kmeans`` (~0.275) or Ward (~0.20), but Dasgupta
cost could not discriminate any construction. This script asks the only question
that matters downstream: **does that cophenetic edge cash out as better recovery
of a textbook's actual chapter structure**, and do "Introduction/Conclusion"
chapters really sit near the top of the tree?

It builds ONE labeled point set per book (the field articles most relevant to the
book, each labeled by its argmax-cosine gold chapter) and feeds the SAME points to
three tree constructions:
  1. ``weighted_bisecting_kmeans`` — the repo default recursive-binary-kmeans
     builder (``scalable_clustering.bisecting_kmeans_linkage``, unweighted here).
  2. ``scipy_ward`` — ``linkage(X, method="ward")``.
  3. ``scipy_average_cosine`` — ``linkage(X, method="average", metric="cosine")``.

Per (book, construction) metrics
--------------------------------
* **Dendrogram purity (DP)** — primary. Bottom-up over the tree: for each merge,
  the same-class leaf-pairs whose LCA is that merge are ``cnt_c(L)·cnt_c(R)``, and
  their subtree purity is ``cnt_c(node)/size(node)``. ``DP = Σ_merges Σ_c
  cnt_c(L)·cnt_c(R)·cnt_c(node)/size(node)  /  Σ_c C(n_c,2)``. Reported with a
  random baseline (mean DP over label shuffles).
* **Cophenetic correlation** of THIS per-book tree (``cophenet(Z, pdist(X))``,
  leaf-order aligned) — needed for THE JOIN.
* **Flat-cut ARI / NMI** at ``k = n_chapters`` (``fcluster(Z,k,"maxclust")`` vs
  gold). ARI is compared to the EXISTING recon.json baseline.
* **Intro/Conclusion-height test**: per chapter, generality = (#leaves in the
  smallest subtree containing >=60% of that chapter's works) / (total leaves);
  larger = more spread = nearer the top. Chapters are flagged intro/conclusion by
  title regex or by ``n==1 or n==max``.

THE JOIN (headline deliverables)
--------------------------------
* **A** per-book table: does average-linkage win DP / ARI?
* **B** pooled (4 books x 3 constructions = up to 12 points): corr(cophenetic, DP)
  and corr(cophenetic, ARI). Does the cheap label-free intrinsic metric predict
  the ToC-anchored one?
* **C** intro/conclusion-height verdict pooled across books (sign / rank test of
  flagged-vs-body generality).
* **D** plain-language verdict on whether curriculum-as-path is salvageable with a
  better dendrogram.

CORPUS / SCOPE CAVEAT (important)
--------------------------------
The 4 books were NOT all ingested at subfield 1702. Their close-loop gap JSONs
declare:  cyberbullying -> subfield 1702;  mcs / neuromorphic / color_image ->
FIELD 17. To keep each book's labeled point set comparable to the EXISTING
``cl_<slug>_recon.json`` ARI baseline (computed on that same corpus), this script
loads each book at the scope ITS OWN gap JSON declares — i.e. it reuses the recon
recipe (``CurriculumConfig`` -> ``instantiate(cfg.experiment.cloud)`` with
``filter_pair.pre.{subfield,field}`` -> ``SubCloud(work_types=("article",))`` ->
``embeddings()``/``work_ids()``), exactly the path ``book_toc_reconstruction.py``
uses. The default cloud denoiser is NONE, so embeddings are raw GIST-small (384d),
matching Track 1. The per-book scope is recorded in the output.

CIRCULARITY CAVEAT
------------------
Work labels are assigned by argmax-cosine to the SAME-space chapter embeddings, so
the ABSOLUTE DP is inflated (the labeling and the tree share a geometry). But the
COMPARISON across the three constructions and the JOIN correlation (B) are fair —
all three see identical labeled points, and cophenetic is computed on the same X.

Regenerate (do not edit generated JSON by hand):

  .conda/bin/python -m artifacts_scripts.curriculum.dendro_eval.toc_recovery
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import time
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from numpy.typing import NDArray
from omegaconf import OmegaConf
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from artifacts_scripts.curriculum.config import CurriculumConfig
from artifacts_scripts.curriculum.scalable_clustering import bisecting_kmeans_linkage
from bookgap.embedder import SentenceTransformerEmbedder
from bookgap.subcloud import SubCloud
from bookgap.util import Filter, LOGGER, setup_logging

_MODULE = "artifacts_scripts.curriculum.dendro_eval.toc_recovery"
_HERE = Path(__file__).resolve().parent
_BOOK_GAPS = Path("artifacts/book_gaps")

# (book_id, slug, n_chapters, toc_json, gap_json, recon_json)
_BOOKS = [
    ("W2898513995", "cyberbullying", 8,
     "W2898513995_toc.json", "W2898513995_cyberbullying_ai.json", "cl_cyberbullying_recon.json"),
    ("W4235013513", "mcs", 10,
     "W4235013513_toc.json", "mcs_gap.json", "cl_mcs_recon.json"),
    ("W4247813589", "neuromorphic", 13,
     "W4247813589_toc.json", "neuromorphic_gap.json", "cl_neuromorphic_recon.json"),
    ("W4211017872", "color_image", 24,
     "W4211017872_toc.json", "color_image_gap.json", "cl_color_image_recon.json"),
]

# Intro/Conclusion/overview-style chapter title regex (case-insensitive).
_GENERALITY_RE = re.compile(
    r"(?i)(introduc|overview|preliminar|background|fundamental|"
    r"conclu|summary|future|outlook|epilogue|perspectiv)"
)


def _provenance() -> str:
    cmd = f"python -m {_MODULE} " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    return f"Generated by `{cmd}`. Regenerate with the command above; do not edit by hand."


def _unit(x: NDArray) -> NDArray:
    """L2-normalize rows (idempotent for GIST-small; matches the curriculum pattern)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------
# Corpus loading (recon recipe — same scope each book's gap JSON declares)
# --------------------------------------------------------------------------
def _load_article_space(subfield: str | None, field: str | None):
    """Build the article SubCloud at the requested scope; return (work_ids, emb, id2row).

    Mirrors ``book_toc_reconstruction.main`` exactly so the point population is the
    same corpus the close-loop recon scored. Embeddings are raw (denoiser=NONE).
    """
    cfg = OmegaConf.structured(CurriculumConfig())
    if subfield:
        cfg.experiment.cloud.filter_pair.pre.subfield = str(subfield)
    if field:
        cfg.experiment.cloud.filter_pair.pre.field = str(field)
    cloud = instantiate(cfg.experiment.cloud)
    articles = SubCloud(parent=cloud, post_filter=Filter(work_types=("article",)))
    art_ids = list(articles.work_ids())
    art_emb = np.asarray(articles.embeddings(), dtype=np.float32)
    id2row = {a: i for i, a in enumerate(art_ids)}
    return art_ids, art_emb, id2row


def _embed_chapters(chapters: list[dict]) -> NDArray:
    """Embed gold chapters (title, description) with the SAME GIST-small embedder."""
    with SentenceTransformerEmbedder(SentenceTransformerEmbedder.ModelName.GIST_SMALL) as e:
        chap_emb = e.embed_texts([(c["title"], c.get("description", "")) for c in chapters])
    return np.asarray(chap_emb, dtype=np.float32)


# --------------------------------------------------------------------------
# Tree builders (the same 3 constructions as Track 1)
# --------------------------------------------------------------------------
def _build_bisecting_kmeans(X: NDArray, seed: int) -> tuple[NDArray, NDArray]:
    """Repo default builder. Returns (Z, X_leaf) where X_leaf[leaf_id]=X[orig point].

    bisecting-kmeans assigns leaf IDs in traversal order, not original row order,
    so we remap X into leaf order (a permutation when min_cluster_size=1) — this is
    required for ``cophenet``/``pdist`` and DP leaf indexing to align.
    """
    n = X.shape[0]
    bk = bisecting_kmeans_linkage(
        X.astype(np.float32), min_cluster_size=1, max_depth=None,
        random_state=seed, use_minibatch=False, verbose=False,
    )
    Z = bk.linkage
    lca = bk.leaf_cluster_assignments  # point -> leaf_id; permutation when msc=1
    assert sorted(lca.tolist()) == list(range(n)), "leaf assignments not a permutation"
    inv = np.empty(bk.n_leaf_clusters, dtype=int)
    inv[lca] = np.arange(n)
    return Z, X[inv]


def _build_trees(X: NDArray, seed: int) -> dict[str, tuple[NDArray, NDArray]]:
    """Return {name: (Z, X_leaf)} for the 3 constructions. X is unit-norm."""
    Xd = X.astype(np.float64)
    out: dict[str, tuple[NDArray, NDArray]] = {}
    Zbk, Xleaf_bk = _build_bisecting_kmeans(X, seed)
    out["weighted_bisecting_kmeans"] = (Zbk, Xleaf_bk)
    out["scipy_ward"] = (linkage(Xd, method="ward"), Xd)
    out["scipy_average_cosine"] = (linkage(Xd, method="average", metric="cosine"), Xd)
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _node_members_bottomup(Z: NDArray, n: int):
    """Yield (node_id, member_index_list) per merge, bottom-up. Leaves index X rows."""
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    for m, row in enumerate(Z):
        a, b = int(row[0]), int(row[1])
        la, lb = members[a], members[b]
        merged = la + lb
        members[n + m] = merged
        del members[a]
        del members[b]
        yield a, la, b, lb, n + m, merged


def dendrogram_purity(Z: NDArray, labels: NDArray) -> float:
    """Dendrogram purity via the bottom-up class-count-vector recurrence.

    DP = [ Σ_merges Σ_classes cnt_c(L)·cnt_c(R)·(cnt_c(node)/size(node)) ]
         / [ Σ_classes C(n_c, 2) ]

    For a merge joining subtrees L, R into ``node``, the same-class pairs whose LCA
    is exactly this merge are cnt_c(L)·cnt_c(R) (one leaf from each side), and the
    purity of their LCA subtree w.r.t. class c is cnt_c(node)/size(node). Summing
    over merges and classes and normalizing by the total number of same-class pairs
    gives the standard expected-purity-of-a-random-same-class-pair-LCA. O(merges x
    classes); fine at n ~ a few hundred.
    """
    n = len(labels)
    classes, y = np.unique(labels, return_counts=True)
    cls_index = {c: k for k, c in enumerate(classes)}
    yk = np.array([cls_index[c] for c in labels])  # 0..C-1 per leaf
    C = len(classes)
    denom = float(np.sum(y * (y - 1) / 2.0))
    if denom <= 0:
        return float("nan")

    def cnt_vec(idxs: list[int]) -> NDArray:
        v = np.zeros(C, dtype=np.float64)
        np.add.at(v, yk[idxs], 1.0)
        return v

    num = 0.0
    for _a, la, _b, lb, _node, merged in _node_members_bottomup(Z, n):
        cnt_l = cnt_vec(la)
        cnt_r = cnt_vec(lb)
        cnt_node = cnt_l + cnt_r
        size_node = float(len(merged))
        # per class: pairs created here * subtree purity for that class
        num += float(np.sum(cnt_l * cnt_r * (cnt_node / size_node)))
    return num / denom


def dendrogram_purity_random(Z: NDArray, labels: NDArray, n_shuffle: int,
                             rng: np.random.Generator) -> float:
    """Mean DP over ``n_shuffle`` label permutations (fixed tree, shuffled labels)."""
    vals = []
    for _ in range(n_shuffle):
        perm = rng.permutation(len(labels))
        vals.append(dendrogram_purity(Z, labels[perm]))
    return float(np.mean(vals))


def chapter_generality(Z: NDArray, labels: NDArray, chapter_label, n: int,
                       frac: float = 0.60) -> float:
    """Generality of one chapter = (size of smallest subtree containing >=frac of the
    chapter's works) / n. Larger = the chapter's works only co-locate high in the
    tree = it's spread across the corpus = nearer the top.

    Bottom-up: the first (smallest) node whose member set covers >=frac of the
    chapter's leaves is the answer; if no internal node reaches it before the root,
    the root (size n) does, giving generality 1.0.
    """
    chap_leaves = set(np.flatnonzero(labels == chapter_label).tolist())
    need = int(np.ceil(frac * len(chap_leaves)))
    if need <= 0:
        return float("nan")
    if need == 1:
        # a single leaf already satisfies >=frac; smallest containing subtree is a
        # leaf (size 1) -> generality 1/n. (Only happens for 1-2 work chapters.)
        return 1.0 / n
    best = n  # root fallback
    for _a, _la, _b, _lb, _node, merged in _node_members_bottomup(Z, n):
        c = sum(1 for i in merged if i in chap_leaves)
        if c >= need:
            best = min(best, len(merged))
            break  # merges are size-monotone bottom-up; first hit is smallest
    return best / n


def _eval_construction(name: str, Z: NDArray, X_leaf: NDArray, labels_leaf: NDArray,
                       k: int, n_shuffle: int, rng: np.random.Generator) -> dict:
    """All per-(book,construction) metrics. ``labels_leaf`` aligned to X_leaf rows."""
    n = X_leaf.shape[0]
    dist = pdist(X_leaf)
    coph = float(cophenet(Z, dist)[0])
    dp = dendrogram_purity(Z, labels_leaf)
    dp_rand = dendrogram_purity_random(Z, labels_leaf, n_shuffle, rng)
    flat = fcluster(Z, t=k, criterion="maxclust")
    k_eff = int(len(np.unique(flat)))
    ari = float(adjusted_rand_score(labels_leaf, flat))
    nmi = float(normalized_mutual_info_score(labels_leaf, flat))
    return {
        "construction": name,
        "dendrogram_purity": dp,
        "dp_above_random": dp - dp_rand,
        "dp_random_baseline": dp_rand,
        "cophenetic_corr": coph,
        "ari_at_k": ari,
        "nmi_at_k": nmi,
        "flat_k_requested": k,
        "flat_k_effective": k_eff,
    }


# --------------------------------------------------------------------------
# Per-book driver
# --------------------------------------------------------------------------
def _process_book(book_id: str, slug: str, n_chapters: int, toc_name: str,
                  gap_name: str, recon_name: str, top_n_mode: str,
                  n_shuffle: int, seed: int) -> dict | None:
    toc_path = _BOOK_GAPS / toc_name
    gap_path = _BOOK_GAPS / gap_name
    recon_path = _BOOK_GAPS / recon_name
    if not toc_path.exists():
        LOGGER.warning("[%s] SKIP: missing ToC %s", slug, toc_path)
        return {"slug": slug, "book_id": book_id, "skipped": f"missing {toc_path}"}

    toc = json.loads(toc_path.read_text())
    chapters = toc["chapters"]
    n_chap = len(chapters)

    # scope from gap JSON (the corpus the recon scored)
    subfield = field = None
    n_gap_articles = None
    if gap_path.exists():
        gap = json.loads(gap_path.read_text())
        subfield = gap.get("scope_subfield")
        field = gap.get("scope_field")
        bk = gap.get("book", gap)
        ga = bk.get("gap_article_ids")
        n_gap_articles = len(ga) if ga else None
    recon = json.loads(recon_path.read_text()) if recon_path.exists() else {}
    ari_recon = recon.get("adjusted_rand_index")
    if n_gap_articles is None:
        n_gap_articles = recon.get("n_gap_articles")

    LOGGER.info("[%s] book_id=%s n_chap=%d scope(subfield=%s field=%s) "
                "n_gap_articles=%s ari_recon=%s",
                slug, book_id, n_chap, subfield, field, n_gap_articles, ari_recon)

    try:
        art_ids, art_emb, _id2row = _load_article_space(subfield, field)
    except Exception as exc:  # noqa: BLE001 — skip-and-proceed per spec
        LOGGER.warning("[%s] SKIP: corpus load failed: %s", slug, exc)
        return {"slug": slug, "book_id": book_id, "skipped": f"corpus load failed: {exc}"}

    LOGGER.info("[%s] article pool: %d works, dim=%d", slug, len(art_ids), art_emb.shape[1])

    # --- labeled point set: top-N by max-cosine to ANY chapter, argmax-cosine label.
    # The article pool can be millions of rows, so keep the scan in float32 and only
    # promote the selected N rows to float64. GIST-small rows are already ~unit, but
    # normalize the (small) chapter matrix and the (small) selected slice anyway so
    # the dot products are exact cosines.
    chap_emb = _unit(_embed_chapters(chapters))                       # (n_chap, d) f64
    chap_f32 = chap_emb.astype(np.float32)
    # row-normalize the pool in float32 in place (cheap; matches cosine on selection)
    art_norm = art_emb / np.maximum(
        np.linalg.norm(art_emb, axis=1, keepdims=True), np.float32(1e-12))
    sims_all = art_norm @ chap_f32.T                                  # (N_pool, n_chap) f32
    max_sim = sims_all.max(axis=1)                                    # best chapter cos
    if top_n_mode == "n_gap_articles" and n_gap_articles:
        N = int(n_gap_articles)
    else:
        N = 60 * n_chap
    N = min(N, len(art_ids))
    top_idx = np.argpartition(-max_sim, N - 1)[:N] if N < len(max_sim) else np.arange(len(max_sim))
    top_idx = np.sort(top_idx)                                        # stable point order
    X = _unit(art_emb[top_idx])                                       # (N, d) unit-norm f64
    labels = sims_all[top_idx].argmax(axis=1).astype(int)            # gold chapter per work
    n = X.shape[0]
    label_sizes = np.bincount(labels, minlength=n_chap)
    LOGGER.info("[%s] labeled point set: N=%d (mode=%s), chapter sizes=%s",
                slug, n, top_n_mode if (top_n_mode == "n_gap_articles" and n_gap_articles) else "60xnchap",
                {chapters[i]["title"][:24]: int(label_sizes[i]) for i in range(n_chap)})

    k = n_chap  # flat cut at the number of gold chapters

    # --- build all 3 trees on the SAME X; evaluate
    trees = _build_trees(X, seed)
    constructions: list[dict] = []
    # cache per-construction (Z, X_leaf, labels_leaf) for the generality test
    leaf_cache: dict[str, tuple[NDArray, NDArray, NDArray]] = {}
    for name, (Z, X_leaf) in trees.items():
        # labels for the leaf-ordered X: bisecting-kmeans permutes rows, so recover
        # each leaf's label by matching X_leaf rows back to X rows.
        if name == "weighted_bisecting_kmeans":
            # X_leaf[leaf]=X[inv[leaf]]; reconstruct the permutation by exact match.
            # (X rows are unit and distinct enough; use argmax dot = identity match.)
            sim = X_leaf @ X.T
            perm = sim.argmax(axis=1)
            assert len(set(perm.tolist())) == n, "leaf->row map not a bijection"
            labels_leaf = labels[perm]
        else:
            labels_leaf = labels
        leaf_cache[name] = (Z, X_leaf, labels_leaf)
        rng = np.random.default_rng(seed)
        res = _eval_construction(name, Z, X_leaf, labels_leaf, k, n_shuffle, rng)
        if ari_recon is not None:
            res["ari_recon_baseline"] = float(ari_recon)
        constructions.append(res)
        LOGGER.info("[%s] %-26s DP=%.4f (rand %.4f, +%.4f)  coph=%.4f  ARI@%d=%.4f  NMI=%.4f",
                    slug, name, res["dendrogram_purity"], res["dp_random_baseline"],
                    res["dp_above_random"], res["cophenetic_corr"], k, res["ari_at_k"],
                    res["nmi_at_k"])

    # --- intro/conclusion-height test (per construction; report per-chapter generality)
    height: dict[str, dict] = {}
    for name, (Z, X_leaf, labels_leaf) in leaf_cache.items():
        per_chap = []
        for ci in range(n_chap):
            if label_sizes[ci] == 0:
                continue
            title = chapters[ci]["title"]
            cn = chapters[ci]["n"]
            flagged = bool(_GENERALITY_RE.search(title)) or cn == 1 or cn == n_chap
            g = chapter_generality(Z, labels_leaf, ci, n)
            per_chap.append({"chapter_n": cn, "chapter": title, "n_works": int(label_sizes[ci]),
                             "flagged": flagged, "generality": g})
        # rank by generality (descending = nearer top first)
        per_chap.sort(key=lambda d: -d["generality"])
        m = len(per_chap)
        for rank, d in enumerate(per_chap):
            d["generality_rank"] = rank + 1  # 1 = most general (top of tree)
            d["in_top_third"] = (rank + 1) <= max(1, m // 3)
        gen_flagged = [d["generality"] for d in per_chap if d["flagged"]]
        gen_body = [d["generality"] for d in per_chap if not d["flagged"]]
        height[name] = {
            "per_chapter": per_chap,
            "n_flagged": len(gen_flagged),
            "n_body": len(gen_body),
            "mean_generality_flagged": float(np.mean(gen_flagged)) if gen_flagged else float("nan"),
            "mean_generality_body": float(np.mean(gen_body)) if gen_body else float("nan"),
            "flagged_in_top_third_frac": (
                float(np.mean([d["in_top_third"] for d in per_chap if d["flagged"]]))
                if gen_flagged else float("nan")
            ),
        }

    return {
        "slug": slug,
        "book_id": book_id,
        "scope": {"subfield": subfield, "field": field},
        "n_chapters": n_chap,
        "n_article_pool": len(art_ids),
        "n_labeled_points": n,
        "top_n": int(N),
        "top_n_mode": (top_n_mode if (top_n_mode == "n_gap_articles" and n_gap_articles) else "60xnchap"),
        "chapter_label_sizes": {chapters[i]["title"]: int(label_sizes[i]) for i in range(n_chap)},
        "ari_recon_baseline": float(ari_recon) if ari_recon is not None else None,
        "constructions": constructions,
        "height_test": height,
    }


# --------------------------------------------------------------------------
# THE JOIN + reporting
# --------------------------------------------------------------------------
def _print_table_A(books: list[dict]) -> None:
    print("\n" + "=" * 110)
    print("TABLE A — per-(book x construction) recovery")
    print("=" * 110)
    hdr = (f"{'book':<13}{'construction':<26}{'DP':>8}{'DP-rand':>9}{'DP>rand':>9}"
           f"{'coph':>8}{'ARI@k':>8}{'ARI_rec':>9}{'NMI':>8}")
    print(hdr)
    print("-" * len(hdr))
    for b in books:
        if b.get("skipped"):
            print(f"{b['slug']:<13}SKIPPED: {b['skipped']}")
            continue
        for i, r in enumerate(b["constructions"]):
            bn = b["slug"] if i == 0 else ""
            arr = b.get("ari_recon_baseline")
            arr_s = f"{arr:.4f}" if arr is not None else "  n/a"
            print(f"{bn:<13}{r['construction']:<26}{r['dendrogram_purity']:>8.4f}"
                  f"{r['dp_random_baseline']:>9.4f}{r['dp_above_random']:>9.4f}"
                  f"{r['cophenetic_corr']:>8.4f}{r['ari_at_k']:>8.4f}{arr_s:>9}"
                  f"{r['nmi_at_k']:>8.4f}")
        print("-" * len(hdr))


def _print_table_B(books: list[dict]) -> dict:
    """Pool (book x construction) points; correlate cophenetic vs DP and vs ARI."""
    coph, dp, ari, dp_rel = [], [], [], []
    rows = []
    for b in books:
        if b.get("skipped"):
            continue
        for r in b["constructions"]:
            coph.append(r["cophenetic_corr"])
            dp.append(r["dendrogram_purity"])
            dp_rel.append(r["dp_above_random"])
            ari.append(r["ari_at_k"])
            rows.append((b["slug"], r["construction"], r["cophenetic_corr"],
                         r["dendrogram_purity"], r["ari_at_k"]))
    out: dict = {"n_points": len(coph)}
    print("\n" + "=" * 78)
    print(f"TABLE B — THE JOIN: does cophenetic (label-free) predict ToC recovery?  "
          f"(n={len(coph)} points)")
    print("=" * 78)
    if len(coph) >= 3:
        def _corr(x, y):
            x = np.asarray(x); y = np.asarray(y)
            pr = pearsonr(x, y); sr = spearmanr(x, y)
            return float(pr.statistic), float(pr.pvalue), float(sr.statistic), float(sr.pvalue)
        for label, ys, key in [("DP", dp, "dp"), ("DP>rand", dp_rel, "dp_above_random"),
                               ("ARI@k", ari, "ari")]:
            pr, pp, sr, sp = _corr(coph, ys)
            out[f"coph_vs_{key}"] = {"pearson_r": pr, "pearson_p": pp,
                                     "spearman_r": sr, "spearman_p": sp}
            print(f"  cophenetic vs {label:<9}  Pearson r={pr:+.3f} (p={pp:.3f})   "
                  f"Spearman rho={sr:+.3f} (p={sp:.3f})")
    else:
        print("  too few points for correlation")
    print("\n  pooled points (slug, construction, coph, DP, ARI@k):")
    for s, c, cp, d, a in rows:
        print(f"    {s:<13}{c:<26}coph={cp:+.3f}  DP={d:.3f}  ARI={a:+.3f}")
    out["points"] = [{"slug": s, "construction": c, "cophenetic": cp, "dp": d, "ari": a}
                     for s, c, cp, d, a in rows]
    return out


def _print_table_C(books: list[dict]) -> dict:
    """Intro/conclusion-height verdict pooled across books (per construction).

    Per book, per construction: a (mean_generality_flagged, mean_generality_body)
    pair and the flagged-in-top-third fraction. Pool the within-book paired
    differences (flagged - body) and run a sign / Wilcoxon test per construction.
    """
    print("\n" + "=" * 86)
    print("TABLE C — intro/conclusion-height: do flagged chapters sit nearer the TOP?")
    print("=" * 86)
    by_constr: dict[str, list[tuple[str, float, float, float]]] = {}
    for b in books:
        if b.get("skipped"):
            continue
        for name, h in b["height_test"].items():
            by_constr.setdefault(name, []).append(
                (b["slug"], h["mean_generality_flagged"], h["mean_generality_body"],
                 h["flagged_in_top_third_frac"]))
    out: dict = {}
    for name, rows in by_constr.items():
        diffs = [(f - bd) for (_s, f, bd, _t) in rows
                 if not (np.isnan(f) or np.isnan(bd))]
        top3 = [t for (_s, _f, _bd, t) in rows if not np.isnan(t)]
        n_pos = sum(1 for d in diffs if d > 0)
        sign_p = float("nan")
        wil_p = float("nan")
        if len(diffs) >= 1:
            # exact two-sided sign test p-value
            from math import comb
            npos, ntot = n_pos, len(diffs)
            k_ext = max(npos, ntot - npos)
            tail = sum(comb(ntot, j) for j in range(k_ext, ntot + 1))
            sign_p = float(min(1.0, 2.0 * tail / (2 ** ntot)))
        if len(diffs) >= 3 and len(set(np.sign(diffs))) >= 1 and any(d != 0 for d in diffs):
            try:
                wil_p = float(wilcoxon(diffs).pvalue)
            except ValueError:
                wil_p = float("nan")
        out[name] = {
            "n_books": len(rows),
            "mean_diff_flagged_minus_body": float(np.mean(diffs)) if diffs else float("nan"),
            "n_books_flagged_more_general": n_pos,
            "sign_test_p": sign_p,
            "wilcoxon_p": wil_p,
            "mean_flagged_in_top_third_frac": float(np.mean(top3)) if top3 else float("nan"),
            "per_book": [{"slug": s, "gen_flagged": f, "gen_body": bd, "top_third_frac": t}
                         for (s, f, bd, t) in rows],
        }
        print(f"\n  [{name}]")
        print(f"    books where flagged MORE general (higher) than body: "
              f"{n_pos}/{len(diffs)}  (sign p={sign_p:.3f}, wilcoxon p={wil_p:.3f})")
        print(f"    mean (gen_flagged - gen_body) = {out[name]['mean_diff_flagged_minus_body']:+.4f}  "
              f"[positive => flagged nearer top]")
        print(f"    mean fraction of flagged chapters landing in TOP THIRD = "
              f"{out[name]['mean_flagged_in_top_third_frac']:.3f}")
        for s, f, bd, t in rows:
            mark = "  ✓" if (not np.isnan(f) and not np.isnan(bd) and f > bd) else "  ✗"
            print(f"      {s:<13} flagged={f:.3f}  body={bd:.3f}  top3frac={t:.3f}{mark}")
    return out


def _verdict_D(books: list[dict], joinB: dict, heightC: dict) -> dict:
    """Plain-language synthesis of the motivating question."""
    live = [b for b in books if not b.get("skipped")]
    # (i) is any construction meaningfully better at recovery?
    best_dp_by_book = {}
    avg_win = {"weighted_bisecting_kmeans": 0, "scipy_ward": 0, "scipy_average_cosine": 0}
    ari_win = dict(avg_win)
    for b in live:
        dps = {r["construction"]: float(r["dendrogram_purity"]) for r in b["constructions"]}
        aris = {r["construction"]: float(r["ari_at_k"]) for r in b["constructions"]}
        best_dp = max(dps, key=lambda c: dps[c])
        best_dp_by_book[b["slug"]] = best_dp
        avg_win[best_dp] += 1
        ari_win[max(aris, key=lambda c: aris[c])] += 1
    # (ii) is the ceiling low for ALL? -> DP barely above random everywhere
    all_dp_above = [r["dp_above_random"] for b in live for r in b["constructions"]]
    all_ari = [r["ari_at_k"] for b in live for r in b["constructions"]]
    # (iii) intro/conclusion mechanism
    avg_constr = heightC.get("scipy_average_cosine", {})
    lines = []
    lines.append(f"Books evaluated: {[b['slug'] for b in live]} "
                 f"({len(live)}/{len(books)}).")
    lines.append(f"(i) DP winner per book: {best_dp_by_book}. "
                 f"DP-win tally across books: {avg_win}; ARI-win tally: {ari_win}.")
    cv = joinB.get("coph_vs_dp", {})
    cva = joinB.get("coph_vs_ari", {})
    lines.append(f"    JOIN: corr(cophenetic, DP) Pearson r="
                 f"{cv.get('pearson_r', float('nan')):+.3f} (p={cv.get('pearson_p', float('nan')):.3f}); "
                 f"corr(cophenetic, ARI) r={cva.get('pearson_r', float('nan')):+.3f} "
                 f"(p={cva.get('pearson_p', float('nan')):.3f}).")
    lines.append(f"(ii) DP-above-random across all (book x construction): "
                 f"min={np.min(all_dp_above):+.4f} mean={np.mean(all_dp_above):+.4f} "
                 f"max={np.max(all_dp_above):+.4f}.  ARI@k: min={np.min(all_ari):+.3f} "
                 f"mean={np.mean(all_ari):+.3f} max={np.max(all_ari):+.3f}.")
    lines.append(f"(iii) intro/conclusion-near-top (avg-linkage tree): flagged-more-general "
                 f"in {avg_constr.get('n_books_flagged_more_general', '?')}/"
                 f"{avg_constr.get('n_books', '?')} books "
                 f"(sign p={avg_constr.get('sign_test_p', float('nan')):.3f}); "
                 f"mean(flagged-body)={avg_constr.get('mean_diff_flagged_minus_body', float('nan')):+.4f}; "
                 f"flagged-in-top-third frac={avg_constr.get('mean_flagged_in_top_third_frac', float('nan')):.3f}.")
    return {"lines": lines, "dp_win_tally": avg_win, "ari_win_tally": ari_win,
            "dp_above_random_minmeanmax": [float(np.min(all_dp_above)),
                                           float(np.mean(all_dp_above)), float(np.max(all_dp_above))],
            "ari_minmeanmax": [float(np.min(all_ari)), float(np.mean(all_ari)),
                               float(np.max(all_ari))]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-n-mode", choices=["n_gap_articles", "60xnchap"],
                    default="n_gap_articles",
                    help="labeled point count: recon's n_gap_articles (default) or 60*n_chapters.")
    ap.add_argument("--n-shuffle", type=int, default=5,
                    help="label shuffles for the DP random baseline (default 5).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(_HERE / "toc_recovery_results.json"))
    args = ap.parse_args()

    setup_logging(log_name="toc_recovery", log_level="INFO", include_time=False,
                  log_path=Path(args.out).with_suffix(".log"))
    t0 = time.time()

    books: list[dict] = []
    for book_id, slug, n_chap, toc_name, gap_name, recon_name in _BOOKS:
        LOGGER.info("=" * 70)
        try:
            res = _process_book(book_id, slug, n_chap, toc_name, gap_name, recon_name,
                                args.top_n_mode, args.n_shuffle, args.seed)
        except Exception as exc:  # noqa: BLE001 — skip-and-proceed per spec
            LOGGER.exception("[%s] unexpected failure; skipping", slug)
            res = {"slug": slug, "book_id": book_id, "skipped": f"unexpected: {exc}"}
        if res is not None:
            books.append(res)

    # --- tables + verdict
    _print_table_A(books)
    joinB = _print_table_B(books)
    heightC = _print_table_C(books)
    verdictD = _verdict_D(books, joinB, heightC)

    print("\n" + "=" * 86)
    print("VERDICT D — is curriculum-as-path salvageable with a better dendrogram?")
    print("=" * 86)
    for ln in verdictD["lines"]:
        print("  " + ln)

    out = {
        "books": books,
        "join_B": joinB,
        "height_C": heightC,
        "verdict_D": verdictD,
        "params": {"top_n_mode": args.top_n_mode, "n_shuffle": args.n_shuffle,
                   "seed": args.seed},
        "caveats": {
            "scope": ("cyberbullying loaded at subfield 1702; mcs/neuromorphic/color_image "
                      "at FIELD 17 — each at the scope its own gap JSON declares, matching "
                      "the cl_<slug>_recon.json corpus. Default cloud denoiser=NONE (raw GIST-small)."),
            "circularity": ("work labels = argmax-cosine to same-space chapter embeddings, so "
                            "ABSOLUTE DP is inflated; the cross-construction COMPARISON and the "
                            "JOIN correlation (B) are fair (identical labeled points; coph on same X)."),
        },
        "elapsed_s": round(time.time() - t0, 1),
        "provenance": _provenance(),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[done] {time.time() - t0:.1f}s  results -> {args.out}")


if __name__ == "__main__":
    main()
