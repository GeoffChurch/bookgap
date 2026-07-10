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

    PYTHONPATH=. uv run python -m artifacts_scripts.curriculum.toc_dendrogram_cut \
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


def build_tree(Z: np.ndarray) -> Tuple[Node, Dict[int, Node], Dict[int, int]]:
    import sys

    from scipy.cluster.hierarchy import leaves_list, linkage, to_tree

    sys.setrecursionlimit(max(10000, 4 * Z.shape[0]))
    L = linkage(Z, method="ward")
    leaf_pos = {int(lid): rank for rank, lid in enumerate(leaves_list(L))}  # 1-D order
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

    return conv(root_c), nodes, leaf_pos


def adaptive_cut(root: Node, key: str, Z: np.ndarray, mass: np.ndarray,
                 alpha: float = 0.06, max_n: int = 40) -> List[Node]:
    """Let the data pick N: split the largest cluster (by `key`) while every
    cluster stays >= alpha * equal-mass-share. The balance-floor operationalizes
    'split the top, stop when the smallest would fall' (split-and-evict's intent).
    """
    # Ward top-splits are very unbalanced (one broad cluster + a sliver each level),
    # so a balance-floor degenerates to N=2. Instead split the biggest OVER-CAP
    # cluster until none exceeds the cap -> N* is whatever the field's granularity
    # yields. (alpha = cap as a fraction of the corpus; eviction-merge is a future
    # refinement of the literal split-and-evict.)
    cap = max(8.0, alpha * len(root.members))
    clusters = [root]
    while len(clusters) < max_n:
        cand = [c for c in clusters if c.left is not None and len(c.members) > cap]
        if not cand:
            break
        largest = max(cand, key=lambda c: len(c.members))
        clusters = [c for c in clusters if c is not largest] + [largest.left, largest.right]
    return clusters


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
# Soft / structure-aware matching suite (ST space; cluster in LSA, evaluate here)
# --------------------------------------------------------------------------- #
def st_embed(work_texts: List[str], chapters, model_name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import normalize

    m = SentenceTransformer(model_name, device="cpu")
    ctext = [f"{c.get('title','')}. {c.get('description','')}" for c in chapters]
    W = normalize(m.encode(work_texts, batch_size=64, show_progress_bar=False))
    C = normalize(m.encode(ctext, show_progress_bar=False))
    return W, C


def assign_chapters(W: np.ndarray, C: np.ndarray, mean_center: bool = True) -> List[set]:
    if mean_center:
        mu = W.mean(0, keepdims=True)
        W, C = W - mu, C - mu
    a = (W @ C.T).argmax(1)
    return [set(np.where(a == j)[0].tolist()) for j in range(C.shape[0])]


def topic_centroids(leaves: List[Node], W: np.ndarray, mass: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import normalize

    rows = []
    for c in leaves:
        w = mass[c.members]
        rows.append((W[c.members] * w[:, None]).sum(0) / (w.sum() + 1e-12))
    return normalize(np.array(rows))


def _uot(C: np.ndarray, a: np.ndarray, b: np.ndarray, eps=0.05, rho=1.0, iters=500):
    """Unbalanced Sinkhorn — KL-relaxed marginals so neither side is forced to
    fully match (lets the algo over/under-resolve vs the human ToC)."""
    K = np.exp(-C / eps)
    u, v = np.ones_like(a), np.ones_like(b)
    f = rho / (rho + eps)
    for _ in range(iters):
        u = (a / (K @ v + 1e-300)) ** f
        v = (b / (K.T @ u + 1e-300)) ** f
    return u[:, None] * K * v[None, :]


def uot_match(T: np.ndarray, C_st: np.ndarray, leaves, jsets, mass, chapters):
    """Soft topic<->chapter transport. Returns transported similarity + per-chapter
    'effective #topics feeding it' = exp(entropy of its transport column): high =>
    GENERAL chapter (many topics flow in), ~1 => specific. Reveals intro/survey
    chapters without pre-deciding how many to include."""
    S = T @ C_st.T                       # (nT, M) cosine in ST space
    a = np.array([mass[c.members].sum() for c in leaves], float); a /= a.sum()
    b = np.array([sum(mass[i] for i in s) for s in jsets], float); b /= b.sum()
    P = _uot(1.0 - S, a, b)
    transported = float((P * S).sum() / P.sum())
    gen = []
    for j in range(len(jsets)):
        col = P[:, j] / (P[:, j].sum() + 1e-12)
        ent = float(-(col * np.log(col + 1e-12)).sum())
        gen.append((chapters[j].get("n"), chapters[j].get("title", ""), math.exp(ent)))
    return transported, gen


def path_likeness(T: np.ndarray, C_st: np.ndarray, leaves, leaf_pos, chapters):
    """Map each chapter to its nearest topic; is the chapter sequence monotone in
    the dendrogram's 1-D leaf order? High |Spearman| => the ToC traces a path."""
    from scipy.stats import spearmanr

    tpos = np.array([np.mean([leaf_pos[m] for m in c.members]) for c in leaves])
    nearest = (T @ C_st.T).argmax(0)                      # chapter -> topic
    chap_pos = [tpos[nearest[j]] for j in range(len(chapters))]
    rho, p = spearmanr(range(len(chapters)), chap_pos)
    return float(rho), float(p), len(set(nearest.tolist())), nearest


def transition_edges(T: np.ndarray, C_st: np.ndarray, chapters, n_perm=4000, seed=0):
    """The weighted complete graph BEFORE linearization: do Jurafsky's consecutive
    chapters land on high-similarity topic pairs (good edges) vs a random order?"""
    G = T @ T.T                                           # topic-topic similarity
    nearest = (T @ C_st.T).argmax(0)                      # chapter -> topic
    seq = [nearest[j] for j in range(len(chapters))]

    def mean_edge(order):
        w = [G[seq[order[k]], seq[order[k + 1]]]
             for k in range(len(order) - 1) if seq[order[k]] != seq[order[k + 1]]]
        return float(np.mean(w)) if w else float("nan")

    actual = mean_edge(list(range(len(chapters))))
    rng = np.random.default_rng(seed)
    rand = [mean_edge(list(rng.permutation(len(chapters)))) for _ in range(n_perm)]
    pct = float(np.mean([r < actual for r in rand]))
    return actual, float(np.nanmean(rand)), pct


def run_analyses(root, Z, mass, texts, chapters, leaf_pos, N):
    print("ST-embedding works + chapters (neutral space)...")
    W_st, C_st = st_embed(texts, chapters)
    jsets = assign_chapters(W_st, C_st)
    jtitles = [(c.get("n"), c.get("title", "")) for c in chapters]

    adaptive = adaptive_cut(root, "effN_sil", Z, mass)
    cuts = {
        f"effN_sil@{N}": heap_cut(root, "effN_sil", N, Z, mass),
        f"adaptive@{len(adaptive)}": adaptive,
        f"kmeans@{N}": kmeans_cut(Z, N),
    }
    print(f"adaptive split-and-evict picked N* = {len(adaptive)} (vs forced {N})\n")

    gen_for_table = None
    print(f"{'cut':<16}{'UOT-sim':>9}{'path-Spearman':>15}{'distinct/'+str(N):>12}"
          f"{'good-edge pctile':>18}")
    for name, leaves in cuts.items():
        T = topic_centroids(leaves, W_st, mass)
        uot, gen = uot_match(T, C_st, leaves, jsets, mass, chapters)
        rho, p, ndist, _ = path_likeness(T, C_st, leaves, leaf_pos, chapters)
        act, rnd, pct = transition_edges(T, C_st, chapters)
        print(f"{name:<16}{uot:>9.3f}{f'{rho:+.2f} (p={p:.2f})':>15}"
              f"{f'{ndist}/{len(jsets)}':>12}{f'{act:.2f} vs {rnd:.2f} [{pct:.0%}]':>18}")
        if name.startswith("effN_sil"):
            gen_for_table = gen

    # --- general-node recovery: add broad internal nodes as match candidates so
    #     general/overview chapters can match a general node instead of being
    #     force-fit to a specific leaf (the non-antichain / cycle idea, diagnostic) ---
    n_total = len(root.members)
    high, stack = [], [root]
    while stack:
        nd = stack.pop()
        if nd.left is not None and len(nd.members) > 0.15 * n_total:
            high.append(nd)
            stack += [nd.left, nd.right]
    high = sorted(high, key=lambda c: -len(c.members))[:8]
    base = cuts[f"effN_sil@{N}"]
    print(f"\n--- general-node recovery (+{len(high)} broad nodes as candidates) ---")
    for label, cand in [("leaves only", base), ("leaves+general", base + high)]:
        Tc = topic_centroids(cand, W_st, mass)
        nearest = (Tc @ C_st.T).argmax(0)
        routed = sum(1 for j in range(len(jsets)) if nearest[j] >= len(base))
        print(f"  [{label:<14}] distinct candidates hit = {len(set(nearest.tolist()))}"
              f"/{len(jsets)}; chapters routed to a general node = {routed}")
    Tc = topic_centroids(base + high, W_st, mass)
    nearest = (Tc @ C_st.T).argmax(0)
    routed_ch = [jtitles[j] for j in range(len(jsets)) if nearest[j] >= len(base)]
    print("  chapters that PREFER a general node: "
          + ", ".join(f"ch{n}:{str(t)[:18]}" for n, t in routed_ch))

    print("\nMost GENERAL Jurafsky chapters (UOT effective #topics feeding in; "
          "high => overview/intro that a flat partition can't represent):")
    for n, t, g in sorted(gen_for_table, key=lambda x: -x[2])[:6]:
        print(f"  ch{n:>2} {str(t)[:40]:<40} eff#topics={g:.1f}")
    print("Most SPECIFIC (eff#topics ~1 => one topic owns it):")
    for n, t, g in sorted(gen_for_table, key=lambda x: x[2])[:6]:
        print(f"  ch{n:>2} {str(t)[:40]:<40} eff#topics={g:.1f}")


def generate_toc(root, leaves, T, W_st, mass, n_total):
    """Emit an ordered algo ToC: a general INTRO node + the topics in a greedy
    GOOD-EDGE walk (the leaf-order/Euler-tour is null even for Jurafsky, so order
    by local topical adjacency instead). Returns (intro_node, order_indices)."""
    high, stack = [], [root]
    while stack:
        nd = stack.pop()
        if nd.left is not None:
            if 0.10 * n_total <= len(nd.members) <= 0.6 * n_total:
                high.append(nd)
            stack += [nd.left, nd.right]
    if not high:
        high = [root]
    # intro = the broad node most CENTRAL to all topics (max mean similarity to the
    # topic centroids) — a hub overview, not the most populous cluster (size != intro).
    hc = topic_centroids(high, W_st, mass)
    intro = high[int((hc @ T.T).mean(1).argmax())]
    intro_cen = topic_centroids([intro], W_st, mass)[0]
    G = T @ T.T
    start = int((T @ intro_cen).argmax())          # topic nearest the intro overview
    order, rem = [start], set(range(len(T))) - {start}
    while rem:                                       # greedy nearest-unvisited (good-edge) walk
        last = order[-1]
        nxt = max(rem, key=lambda j: G[last, j])
        order.append(nxt)
        rem.discard(nxt)
    return intro, order


def run_generate(root, Z, mass, texts, chapters, leaf_pos, N, X, vocab):
    from scipy.stats import kendalltau

    print("ST-embedding...")
    W_st, C_st = st_embed(texts, chapters)
    jtitles = [(c.get("n"), c.get("title", "")) for c in chapters]
    leaves = heap_cut(root, "effN_sil", N, Z, mass)
    T = topic_centroids(leaves, W_st, mass)
    intro, order = generate_toc(root, leaves, T, W_st, mass, len(root.members))

    print(f"\n=== GENERATED ToC ({N} topics, good-edge order, + intro node) ===")
    intro_cen = topic_centroids([intro], W_st, mass)[0]
    intro_chap = int((C_st @ intro_cen).argmax())
    print(f"  [INTRO/general node, n={len(intro.members)}] {log_odds_terms(X, intro.members, vocab)}")
    print(f"     ↳ nearest human chapter: ch{jtitles[intro_chap][0]} {jtitles[intro_chap][1]}")
    for rank, ti in enumerate(order, 1):
        print(f"  {rank:>2}. {log_odds_terms(X, leaves[ti].members, vocab, 5)}")

    # evaluate the generated ORDER against the human chapter sequence
    G = T @ T.T
    gen_edge = float(np.mean([G[order[k], order[k + 1]] for k in range(len(order) - 1)]))
    nearest = (T @ C_st.T).argmax(0)                 # human chapter -> topic index
    pos = {t: i for i, t in enumerate(order)}
    chap_genpos = [pos[nearest[j]] for j in range(len(chapters))]
    tau, p = kendalltau(range(len(chapters)), chap_genpos)
    # human order's own good-edge weight (consecutive matched topics) for comparison
    hum_edge = float(np.mean([G[nearest[j], nearest[j + 1]]
                              for j in range(len(chapters) - 1)
                              if nearest[j] != nearest[j + 1]]))
    print(f"\n  generated-order good-edge weight = {gen_edge:.3f}; "
          f"human-order good-edge weight = {hum_edge:.3f}")
    print(f"  Kendall-tau(generated order, human order) = {tau:+.2f} (p={p:.3f})")
    print(f"  intro node's nearest chapter = ch{jtitles[intro_chap][0]} "
          f"({'MATCHES human ch1' if jtitles[intro_chap][0] == 1 else 'not ch1'})")


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
    ap.add_argument("--analyses", action="store_true",
                    help="run the soft/structure suite (adaptive-N, UOT, path, edges)")
    ap.add_argument("--generate", action="store_true",
                    help="generate an ordered algo ToC (intro node + good-edge order) + eval")
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
    root, nodes, leaf_pos = build_tree(Z)
    print(f"tree: {len(nodes)} nodes")

    if args.analyses:
        run_analyses(root, Z, mass, texts, chapters, leaf_pos, N)
        return
    if args.generate:
        run_generate(root, Z, mass, texts, chapters, leaf_pos, N, X, vocab)
        return

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
