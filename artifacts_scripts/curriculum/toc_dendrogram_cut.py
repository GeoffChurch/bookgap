"""Adaptive dendrogram heap-cut for ToC-matching — prototype / eyeball harness.

Builds a Ward hierarchy over a field corpus (content space), then cuts it to N
topics with a PLUGGABLE "largest" key on a max-heap (Geoff's formulation):
push root, repeatedly pop the cluster with the largest key and replace it with
its two children until the heap holds N clusters. The key choice is the whole
game:

  height   : merge distance (Ward). Popping by height == the STANDARD horizontal
             dendrogram cut (a flat threshold). Baseline — nothing adaptive.
  mass      : sum of node citation mass. Importance-balanced; splits where the
             field is heavy regardless of internal structure.
  effN      : effective #works = exp(entropy of within-cluster mass). Splits the
             most mass-HETEROGENEOUS cluster, ignores one-mega-paper+dust blobs.
  effN_sil  : effN x child-separability, where separability = silhouette of the
             node's own 2-way child split over its members, mapped to [0,1].
             "Split the cluster that is both internally diverse AND cleanly
             bisectable" — leaves cohesive regions coarse, carves deep where
             there's a real seam. This is the candidate cut for content-natural
             topics (matches how human chapters track joints, not equal mass).

Output: the N algo-topics (size, mass-share, log-odds top terms) for each key,
side by side with the human ToC's chapters, for eyeballing granularity/joints.

Self-contained (numpy/scipy/sklearn). Run with the repo interpreter:

    PYTHONPATH=. .conda/bin/python -m artifacts_scripts.curriculum.toc_dendrogram_cut \
        --corpus cache/oa/nlp_t10181/T10181_dd55aae2 \
        --toc artifacts/book_gaps/jurafsky_toc.json --top-k 4000 --keys height effN_sil
"""

from __future__ import annotations

import argparse
import glob
import gzip
import heapq
import json
import math
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# English function words — abstracts below this fraction are non-English (kills
# the romance-language clusters: "ont, este, les, una").
EN_FUNC = frozenset("""the of and to in is that for with as this are on by an be we it from which
can have has was were our these using used or at not but their they its more than such also between
into when how each both other some may will would about we propose present results method based""".split())

# Register/meta tokens — venue boilerplate that clusters proceedings papers
# together and pollutes labels (added to the English stoplist for embed+label).
REGISTER_STOP = frozenset("""proceedings conference workshop annual meeting volume association
international paper papers et al doi http https dx org pp vol eds ed isbn copyright rights reserved
springer elsevier abstract keywords university department editor editors chapter findings shared task
acl emnlp naacl conll ijcnlp lrec coling wmt eacl tacl aaai ijcai neurips nips icml iclr sp bibtex
""".split())

# Strong front-matter signals — drop the WORK entirely if its abstract carries one.
BOILERPLATE = ("proceedings of the", "this volume", "these proceedings", "acceptance rate",
               "accepted papers", "papers accepted", "doi.org", "dx.doi", "all rights reserved",
               "© ", "this is the proceedings")


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
def _abstract(inv: Optional[dict]) -> str:
    if not inv:
        return ""
    pos: List[Tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            pos.append((i, word))
    pos.sort()
    return " ".join(w for _, w in pos)


def _english_frac(txt: str) -> float:
    toks = txt.lower().split()
    if not toks:
        return 0.0
    return sum(t in EN_FUNC for t in toks) / len(toks)


def load_corpus(corpus_dir: str, top_k: int, clean: bool = True
                ) -> Tuple[List[str], List[str], np.ndarray, List[List[str]]]:
    """Top-k works by citation: returns (ids, texts, mass, referenced_works).

    clean=True drops non-English abstracts (English-function-word fraction < 0.08)
    and proceedings front-matter (BOILERPLATE substrings) BEFORE ranking.
    """
    recs = []
    drop_lang = drop_boiler = 0
    for f in sorted(glob.glob(os.path.join(corpus_dir, "*.json.gz"))):
        page = json.load(gzip.open(f, "rt", encoding="utf-8"))
        for r in page.get("results", []):
            txt = _abstract(r.get("abstract_inverted_index"))
            if len(txt) < 200:  # need enough text to embed
                continue
            if clean:
                low = txt.lower()
                if _english_frac(txt) < 0.08:
                    drop_lang += 1
                    continue
                if any(b in low for b in BOILERPLATE):
                    drop_boiler += 1
                    continue
            recs.append((r["id"], txt, r.get("cited_by_count", 0) or 0,
                         r.get("referenced_works", []) or []))
    if clean:
        print(f"corpus filter: dropped {drop_lang} non-English, {drop_boiler} boilerplate")
    recs.sort(key=lambda x: x[2], reverse=True)
    recs = recs[:top_k]
    ids = [r[0] for r in recs]
    texts = [r[1] for r in recs]
    mass = np.array([float(r[2]) for r in recs])
    refs = [r[3] for r in recs]
    return ids, texts, mass, refs


# --------------------------------------------------------------------------- #
# Embedding (TF-IDF -> LSA, mirrors curriculum_toc's LSA-64 content space)
# --------------------------------------------------------------------------- #
def embed(texts: List[str], dim: int, seed: int = 42, clean: bool = True):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    from sklearn.preprocessing import normalize

    stop = list(ENGLISH_STOP_WORDS.union(REGISTER_STOP)) if clean else "english"
    # alpha-only tokens (>=3 letters) drop years/ordinals/DOI numbers ("2018","52nd","3765")
    pattern = r"(?u)\b[a-z][a-z][a-z]+\b" if clean else r"(?u)\b\w\w+\b"
    vec = TfidfVectorizer(stop_words=stop, token_pattern=pattern, min_df=5, max_df=0.4,
                          ngram_range=(1, 2), sublinear_tf=True, max_features=40000)
    X = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    Z = normalize(svd.fit_transform(X))
    return Z, X, vec.get_feature_names_out(), svd.explained_variance_ratio_.sum(), vec, svd


# --------------------------------------------------------------------------- #
# Hierarchy + heap-cut
# --------------------------------------------------------------------------- #
class Node:
    __slots__ = ("id", "members", "left", "right", "height", "key", "sil")

    def __init__(self, nid, members, left, right, height):
        self.id = nid
        self.members = members      # leaf row indices under this node
        self.left = left
        self.right = right
        self.height = height        # merge distance (0 for leaves)
        self.key = 0.0
        self.sil = float("nan")


def build_tree(Z: np.ndarray) -> Tuple[Node, Dict[int, Node]]:
    import sys

    from scipy.cluster.hierarchy import linkage, to_tree

    sys.setrecursionlimit(max(10000, 4 * Z.shape[0]))
    L = linkage(Z, method="ward")
    root_c, nodes_c = to_tree(L, rd=True)
    nodes: Dict[int, Node] = {}

    def conv(c) -> Node:
        if c.is_leaf():
            n = Node(c.id, [c.id], None, None, 0.0)
        else:
            lft, rgt = conv(c.get_left()), conv(c.get_right())
            n = Node(c.id, lft.members + rgt.members, lft, rgt, c.dist)
        nodes[c.id] = n
        return n

    return conv(root_c), nodes


def child_silhouette(node: Node, Z: np.ndarray) -> float:
    """Silhouette of the node's own 2-way child split over its members, -> [0,1]."""
    if node.left is None or len(node.left.members) < 2 or len(node.right.members) < 2:
        return 0.5  # neutral: can't assess a near-singleton split
    from sklearn.metrics import silhouette_score

    idx = node.members
    lab = np.zeros(len(idx), dtype=int)
    lset = set(node.left.members)
    for i, m in enumerate(idx):
        if m in lset:
            lab[i] = 1
    try:
        s = silhouette_score(Z[idx], lab, metric="euclidean")
    except ValueError:
        return 0.5
    return (1.0 + s) / 2.0  # [-1,1] -> [0,1]


def eff_n(members: List[int], mass: np.ndarray) -> float:
    """Effective #works = exp(entropy) of the within-cluster mass distribution."""
    w = mass[members]
    s = w.sum()
    if s <= 0:
        return float(len(members))
    p = w / s
    p = p[p > 0]
    return float(math.exp(-(p * np.log(p)).sum()))


def compute_key(node: Node, key: str, Z: np.ndarray, mass: np.ndarray) -> float:
    if len(node.members) < 2:
        return -1.0  # never split a singleton
    if key == "height":
        return node.height
    if key == "mass":
        return float(mass[node.members].sum())
    if key == "effN":
        return eff_n(node.members, mass)
    if key == "effN_sil":
        if math.isnan(node.sil):
            node.sil = child_silhouette(node, Z)
        return eff_n(node.members, mass) * node.sil
    raise ValueError(key)


def heap_cut(root: Node, key: str, N: int, Z: np.ndarray, mass: np.ndarray) -> List[Node]:
    """Pop the max-key cluster, replace with its 2 children, until N clusters."""
    def push(h, n):
        n.key = compute_key(n, key, Z, mass)
        heapq.heappush(h, (-n.key, n.id, n))

    heap: List[Tuple[float, int, Node]] = []
    push(heap, root)
    leaves = [root]
    while len(leaves) < N:
        # pop the splittable node with the largest key
        while heap:
            negk, _, node = heapq.heappop(heap)
            if node.left is not None and len(node.members) >= 2:
                break
        else:
            break  # nothing left to split
        leaves.remove(node)
        for child in (node.left, node.right):
            leaves.append(child)
            if child.left is not None:
                push(heap, child)
    return leaves


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def log_odds_terms(X, members: List[int], vocab, k: int = 6) -> str:
    """Top terms by log-odds of in-cluster vs global tf-idf mass."""
    import scipy.sparse as sp

    glob_mass = np.asarray(X.sum(axis=0)).ravel() + 1.0
    sub = X[members]
    sub_mass = np.asarray(sub.sum(axis=0)).ravel() + 1.0
    lo = np.log(sub_mass / sub_mass.sum()) - np.log(glob_mass / glob_mass.sum())
    top = np.argsort(lo)[::-1][:k]
    return ", ".join(vocab[i] for i in top)


# --------------------------------------------------------------------------- #
# Matching against the human ToC (materialize chapters -> work-sets, score)
# --------------------------------------------------------------------------- #
def materialize_toc(chapters, vec, svd, Z, mean_center: bool = True) -> List[set]:
    """Embed each chapter (title+description) into the work space, assign every
    work to its nearest chapter -> the human ToC as a partition of the same works
    (the book_toc_reconstruction bridge; mean-centered to avoid a generic chapter
    hoovering everything)."""
    from sklearn.preprocessing import normalize

    texts = [f"{c.get('title','')} . {c.get('description','')}" for c in chapters]
    C = normalize(svd.transform(vec.transform(texts)))
    W = Z
    if mean_center:
        mu = W.mean(0, keepdims=True)
        W, C = W - mu, C - mu
    assign = (W @ C.T).argmax(1)
    return [set(np.where(assign == j)[0].tolist()) for j in range(len(chapters))]


def materialize_toc_st(chapters, work_texts: List[str], mean_center: bool = True,
                       model_name: str = "all-MiniLM-L6-v2") -> List[set]:
    """De-confounded materialization: assign works to chapters in an INDEPENDENT
    sentence-transformer space (not the LSA-64 geometry the cuts cluster in), so
    KMeans loses its centroid-objective home advantage over the agglomerative cuts."""
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import normalize

    m = SentenceTransformer(model_name, device="cpu")  # don't contend with a resident GPU model
    ctext = [f"{c.get('title','')}. {c.get('description','')}" for c in chapters]
    W = normalize(m.encode(work_texts, batch_size=64, show_progress_bar=False))
    C = normalize(m.encode(ctext, show_progress_bar=False))
    if mean_center:
        mu = W.mean(0, keepdims=True)
        W, C = W - mu, C - mu
    assign = (W @ C.T).argmax(1)
    return [set(np.where(assign == j)[0].tolist()) for j in range(len(chapters))]


def ruzicka(a: set, b: set, mass: np.ndarray) -> float:
    """Mass-weighted Jaccard of two work-sets: Σmin / Σmax."""
    union = a | b
    if not union:
        return 0.0
    inter = a & b
    num = float(sum(mass[i] for i in inter))
    den = float(sum(mass[i] for i in union))
    return num / den if den > 0 else 0.0


def score_cut(leaves: List["Node"], jsets: List[set], mass: np.ndarray,
              jtitles: List[Tuple], label: str) -> Tuple[float, float]:
    """Hungarian 1-1 alignment (mean matched ruzicka) + mass-weighted best-match
    coverage, between the algo topics and the materialized human chapters."""
    from scipy.optimize import linear_sum_assignment

    A = [set(n.members) for n in leaves]
    M = np.zeros((len(A), len(jsets)))
    for i, a in enumerate(A):
        for j, b in enumerate(jsets):
            M[i, j] = ruzicka(a, b, mass)
    ri, cj = linear_sum_assignment(-M)
    matched = [(i, j, M[i, j]) for i, j in zip(ri, cj)]
    mean_matched = float(np.mean([m for _, _, m in matched]))
    tmass = np.array([sum(mass[k] for k in a) for a in A])
    cov = float((np.array([M[i].max() for i in range(len(A))]) * tmass).sum() / tmass.sum())
    print(f"  [{label}] Hungarian mean matched ruzicka = {mean_matched:.3f} | "
          f"mass-wtd best-match coverage = {cov:.3f}")
    for i, j, m in sorted(matched, key=lambda x: -x[2])[:12]:
        print(f"      T{i:<2} <-> ch{jtitles[j][0]:>2} {str(jtitles[j][1])[:36]:<36} ruz={m:.2f}")
    return mean_matched, cov


def kmeans_cut(Z: np.ndarray, N: int, seed: int = 42) -> List["Node"]:
    from sklearn.cluster import KMeans

    lab = KMeans(n_clusters=N, n_init=10, random_state=seed).fit_predict(Z)
    return [Node(-1, np.where(lab == k)[0].tolist(), None, None, 0.0) for k in range(N)]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--toc", required=True)
    ap.add_argument("--top-k", type=int, default=4000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--keys", nargs="+", default=["height", "effN_sil"])
    ap.add_argument("--no-clean", dest="clean", action="store_false",
                    help="disable register/language filtering (show raw debris)")
    ap.add_argument("--materialize", choices=["lsa", "st"], default="lsa",
                    help="reference materialization space: lsa (confounded) or st (neutral)")
    args = ap.parse_args(argv)

    toc = json.load(open(args.toc))
    chapters = toc["chapters"]
    N = len(chapters)
    print(f"=== ToC: {toc['title']} — {N} chapters ===")

    ids, texts, mass, refs = load_corpus(args.corpus, args.top_k, clean=args.clean)
    print(f"corpus: {len(ids)} works (top-{args.top_k} by citation), "
          f"mass range {mass.min():.0f}-{mass.max():.0f}")
    Z, X, vocab, evr, vec, svd = embed(texts, args.dim, clean=args.clean)
    print(f"embed: TF-IDF -> LSA-{args.dim}, EVR {evr:.3f}, vocab {len(vocab)}")
    root, nodes = build_tree(Z)
    print(f"tree: {len(nodes)} nodes")

    if args.materialize == "st":
        jsets = materialize_toc_st(chapters, texts)
    else:
        jsets = materialize_toc(chapters, vec, svd, Z)
    jtitles = [(c.get("n"), c.get("title", "")) for c in chapters]
    print(f"materialized ToC [{args.materialize}]: "
          f"chapter sizes {sorted((len(s) for s in jsets), reverse=True)}\n")

    summary = []
    for key in args.keys:
        leaves = kmeans_cut(Z, N) if key == "kmeans" else heap_cut(root, key, N, Z, mass)
        leaves.sort(key=lambda n: mass[n.members].sum(), reverse=True)
        total = mass.sum()
        print(f"--- cut by '{key}' -> {len(leaves)} topics "
              f"(sizes {min(len(l.members) for l in leaves)}-"
              f"{max(len(l.members) for l in leaves)}) ---")
        for i, n in enumerate(leaves):
            ms = mass[n.members].sum()
            print(f"  T{i:<2} n={len(n.members):<4} mass={ms / total:>5.1%} "
                  f"effN={eff_n(n.members, mass):>5.1f}  {log_odds_terms(X, n.members, vocab)}")
        mm, cov = score_cut(leaves, jsets, mass, jtitles, key)
        summary.append((key, mm, cov))
        print()

    print("=== SCORE SUMMARY (vs materialized human ToC) ===")
    for key, mm, cov in sorted(summary, key=lambda x: -x[1]):
        print(f"  {key:<10} mean-matched-ruzicka {mm:.3f}   coverage {cov:.3f}")

    print(f"--- human ToC chapters ({toc['title']}) ---")
    for c in chapters:
        print(f"  {c.get('n'):>2}. {c.get('title')}")


if __name__ == "__main__":
    main()
