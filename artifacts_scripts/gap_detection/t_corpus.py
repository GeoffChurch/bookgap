"""Track 2 Phase 2: leakage-free T-corpus construction + frozen-transform embedding.

Contract (the leakage guard): every fitted object — TF-IDF vectorizer, LSA/SVD,
keyword selection, KMeans field clustering, scope quantiles — is fit on docs with
creation <= T ONLY. `fit_t_corpus` asserts this and the unit fixture pins it
(a post-T-only token must never enter the fitted vocabulary). Arrivals are
embedded later via the FROZEN T transform (`transform_texts`), never refit.

Sampling design (size-matched to the banked corpus): the T-corpus is a uniform
30k sample of the pre-T eligible pool, constructed as
  part A = the banked 30k subsample's pre-T members (a uniform subsample of the
           pre-T pool, since the banked draw was uniform over the full store)
  part B = a fresh seeded uniform draw from the remaining pre-T pool
so that ALL pre-T banked-fg members are present for the fg-overlap sanity check
while the union stays a uniform sample. Then the banked _build steps replay:
vectorize, drop nnz<5 rows, trim to exactly 30k, SVD-128, KMeans-40 fg selection.

Context hygiene: reads raw abstracts internally; emits aggregates/titles only.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

LSA_DIM = 128
N_KEYWORDS = 2_000
KMEANS_K = 40
FG_SIZE_RANGE = (1_000, 3_000)
MIN_ABSTRACT_CHARS = 200


@dataclass
class TCorpus:
    """Everything Phase 3 needs; fitted objects frozen at T."""

    Z: np.ndarray                 # (n, d) L2-normalized T-LSA coords
    K: sp.csr_matrix              # (n, m) binary doc x keyword (T top-2000-df)
    vocab: np.ndarray             # (m,) keyword strings
    titles: np.ndarray
    page_ids: np.ndarray
    store_rows: np.ndarray        # row index into the 400k store
    days: np.ndarray              # creation days (proxy + exact overrides)
    fg_mask: np.ndarray
    vectorizer: object            # frozen TfidfVectorizer
    svd: object                   # frozen TruncatedSVD
    kmeans: object                # frozen MiniBatchKMeans (40 clusters)
    meta: dict = field(default_factory=dict)

    @property
    def fg_idx(self) -> np.ndarray:
        return np.where(self.fg_mask)[0]

    @property
    def bg_idx(self) -> np.ndarray:
        return np.where(~self.fg_mask)[0]

    def transform_texts(self, texts: list[str]) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
        """FROZEN T-transform: texts -> (Z (L2-normalized LSA), K binary, nnz).

        Rows with nnz < 5 are still returned; callers filter by the returned nnz
        (the corpus eligibility rule) so indices stay aligned with the input.
        """
        X = self.vectorizer.transform(texts)
        nnz = np.asarray(X.getnnz(axis=1)).ravel()
        Zc = self.svd.transform(X).astype(np.float32)
        norms = np.linalg.norm(Zc, axis=1, keepdims=True)
        Z = (Zc / np.maximum(norms, 1e-9)).astype(np.float32)
        kw_cols = self.meta["kw_cols"]
        K = (X[:, kw_cols] > 0).astype(np.uint8).tocsr()
        return Z, K, nnz


def fit_t_corpus(
    texts: list[str],
    days: np.ndarray,
    t_days: float,
    titles: np.ndarray,
    page_ids: np.ndarray,
    store_rows: np.ndarray,
    part_a_rows: np.ndarray,
    seed: int = 42,
    n_docs: int = 30_000,
    fg_size_range: tuple[int, int] = FG_SIZE_RANGE,
    min_interleave: float = 5.0,
    lsa_dim: int = LSA_DIM,
    kmeans_k: int = KMEANS_K,
    n_keywords: int = N_KEYWORDS,
    tfidf_min_df: int = 10,
) -> TCorpus:
    """Fit the full T-corpus pipeline on <=T docs only.

    `texts/days/titles/page_ids/store_rows` are parallel arrays over the PRE-T
    ELIGIBLE POOL (creation <= t_days, abstract >= MIN_ABSTRACT_CHARS) — the
    caller builds the pool; this function ASSERTS the date contract and fits.
    `part_a_rows`: store-row ids that must be force-included (banked pre-T docs).
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize

    days = np.asarray(days, dtype=np.float64)
    if days.size and float(days.max()) > t_days:
        raise ValueError("leakage guard: pool contains post-T docs")
    rng = np.random.default_rng(seed)
    n_pool = len(texts)
    in_a = np.isin(store_rows, part_a_rows)
    n_draw = min(n_docs + 2_000, n_pool)
    rest = np.where(~in_a)[0]
    n_b = max(n_draw - int(in_a.sum()), 0)
    pick_b = rng.choice(rest, size=min(n_b, rest.size), replace=False)
    pick = np.sort(np.concatenate([np.where(in_a)[0], pick_b]))

    vec = TfidfVectorizer(
        stop_words="english", lowercase=True, token_pattern=r"(?u)\b[a-z][a-z]+\b",
        min_df=tfidf_min_df, max_df=0.3, max_features=60_000, sublinear_tf=True,
    )
    X = vec.fit_transform([texts[i] for i in pick])
    keep = np.where(np.asarray(X.getnnz(axis=1)).ravel() >= 5)[0]
    if keep.size > n_docs:
        keep = keep[rng.choice(keep.size, size=n_docs, replace=False)]
        keep.sort()
    X = X[keep]
    pick = pick[keep]

    svd = TruncatedSVD(n_components=min(lsa_dim, X.shape[1] - 1), random_state=seed)
    Z = normalize(svd.fit_transform(X).astype(np.float32)).astype(np.float32)

    df = np.asarray((X > 0).sum(axis=0)).ravel()
    kw_cols = np.argsort(df)[::-1][:n_keywords]
    kw_cols.sort()
    K = (X[:, kw_cols] > 0).astype(np.uint8).tocsr()
    vocab_all = vec.get_feature_names_out()
    vocab = vocab_all[kw_cols]

    km = MiniBatchKMeans(n_clusters=kmeans_k, random_state=seed, n_init=5, batch_size=4096)
    labels = km.fit_predict(Z)
    sizes = np.bincount(labels, minlength=kmeans_k)
    lo, hi = fg_size_range
    candidates = [c for c in range(kmeans_k) if lo <= sizes[c] <= hi]
    if not candidates:
        candidates = [int(np.argmin(np.abs(sizes - (lo + hi) / 2)))]
    diagnostics = []
    for c in candidates:
        members = np.where(labels == c)[0]
        coherence = float(
            np.mean(Z[members] @ km.cluster_centers_[c] / np.linalg.norm(km.cluster_centers_[c]))
        )
        samp = members[rng.choice(members.size, size=min(300, members.size), replace=False)]
        nn_fg = NearestNeighbors(n_neighbors=11, algorithm="brute").fit(Z[members])
        r_ff = nn_fg.kneighbors(Z[samp])[0][:, -1]
        nonmembers = np.where(labels != c)[0]
        nn_bg = NearestNeighbors(n_neighbors=10, algorithm="brute").fit(Z[nonmembers])
        d_bg, _ = nn_bg.kneighbors(Z[samp])
        interleave = float(np.median((d_bg <= r_ff[:, None]).sum(axis=1)))
        diagnostics.append(
            {"cluster": int(c), "size": int(sizes[c]), "coherence": round(coherence, 4),
             "median_bg_within_r10fg": interleave}
        )
    ok = [d for d in diagnostics if d["median_bg_within_r10fg"] >= min_interleave]
    pool_diag = ok if ok else diagnostics
    chosen = max(pool_diag, key=lambda d: d["coherence"])
    fg_mask = labels == chosen["cluster"]

    mean_tfidf = np.asarray(X[fg_mask].mean(axis=0)).ravel()
    top_terms = [str(vocab_all[i]) for i in np.argsort(mean_tfidf)[::-1][:25]]

    meta = {
        "t_days": float(t_days), "seed": seed, "n_docs": int(Z.shape[0]),
        "kmeans_k": int(kmeans_k),
        "n_pool": int(n_pool), "n_part_a": int(in_a.sum()),
        "lsa_dim": LSA_DIM, "tfidf_vocab": int(len(vocab_all)),
        "svd_evr_sum": round(float(svd.explained_variance_ratio_.sum()), 4),
        "fg_cluster": chosen, "fg_top_terms": top_terms,
        "cluster_candidates": diagnostics, "kw_cols": kw_cols,
        "kmeans_labels_chosen": int(chosen["cluster"]),
    }
    return TCorpus(
        Z=Z, K=K, vocab=np.asarray(vocab, dtype=object),
        titles=titles[pick], page_ids=page_ids[pick].astype(np.int64),
        store_rows=store_rows[pick].astype(np.int64), days=days[pick],
        fg_mask=fg_mask, vectorizer=vec, svd=svd, kmeans=km, meta=meta,
    )


def fg_overlap_sanity(tc: TCorpus, banked_fg_page_ids: np.ndarray,
                      labels: np.ndarray | None = None) -> dict:
    """Overlap of the T-fg with banked-fg pre-T members present in the T-corpus.

    Also reports, per T-cluster, where the banked members landed (label
    histogram) so a procedure-pick-vs-field mismatch is visible.
    """
    present = np.isin(tc.page_ids, banked_fg_page_ids)
    n_present = int(present.sum())
    in_fg = int((present & tc.fg_mask).sum())
    lab = tc.kmeans.labels_ if labels is None else labels
    hist = np.bincount(lab[present], minlength=int(tc.meta.get("kmeans_k", KMEANS_K)))
    top = np.argsort(hist)[::-1][:5]
    return {
        "n_banked_fg_present": n_present,
        "n_in_t_fg": in_fg,
        "recall_banked_in_t_fg": round(in_fg / max(n_present, 1), 4),
        "precision_t_fg_is_banked": round(in_fg / max(int(tc.fg_mask.sum()), 1), 4),
        "banked_member_label_hist_top5": [
            {"cluster": int(c), "n": int(hist[c])} for c in top if hist[c] > 0
        ],
        "chosen_cluster": int(tc.meta["kmeans_labels_chosen"]),
    }
