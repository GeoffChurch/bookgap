"""Data assembly for the gap plant-and-recover eval.

Source: cache/wiki/store/wiki_400k/articles.parquet (page_id, title, abstract).
Pipeline (pure sklearn, CPU): seeded 30k subsample -> TF-IDF -> TruncatedSVD(128)
LSA vectors (L2-normalized) + a doc x keyword binary space over the top-2000
terms by document frequency. Foreground = one coherent MiniBatchKMeans cluster
(size 1-3k) chosen for background interleaving; background = the rest.

Context hygiene: this module reads raw abstracts internally but only ever
*emits* aggregates, titles, and numeric diagnostics.

Run:  uv run python -m artifacts_scripts.gap_detection.data
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU is owned by a sweep; CPU only.

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
WIKI_PARQUET = REPO / "cache/wiki/store/wiki_400k/articles.parquet"
OUT_DIR = REPO / "artifacts/gap_detection_eval"
CACHE_DIR = OUT_DIR / "cache"

SEED = 42
N_DOCS = 30_000
LSA_DIM = 128
N_KEYWORDS = 2_000
MIN_ABSTRACT_CHARS = 200
FG_SIZE_RANGE = (1_000, 3_000)
KMEANS_K = 40


@dataclass
class Dataset:
    """Everything downstream needs; no raw text except titles."""

    Z: np.ndarray                      # (n, d) L2-normalized LSA vectors, float32
    K: sp.csr_matrix                   # (n, m) binary doc x keyword
    vocab: np.ndarray                  # (m,) keyword strings
    titles: np.ndarray                 # (n,) title strings
    page_ids: np.ndarray               # (n,) int64
    fg_mask: np.ndarray                # (n,) bool
    meta: dict = field(default_factory=dict)

    @property
    def fg_idx(self) -> np.ndarray:
        return np.where(self.fg_mask)[0]

    @property
    def bg_idx(self) -> np.ndarray:
        return np.where(~self.fg_mask)[0]


def _build(seed: int = SEED, n_docs: int = N_DOCS) -> Dataset:
    import pyarrow.parquet as pq
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize

    rng = np.random.default_rng(seed)
    tbl = pq.read_table(WIKI_PARQUET)
    page_ids = tbl.column("page_id").to_numpy()
    titles = np.asarray(tbl.column("title").to_pylist(), dtype=object)
    abstracts = tbl.column("abstract").to_pylist()
    del tbl

    lengths = np.fromiter((len(a or "") for a in abstracts), dtype=np.int64)
    eligible = np.where(lengths >= MIN_ABSTRACT_CHARS)[0]
    pick = rng.choice(eligible, size=min(n_docs + 2_000, eligible.size), replace=False)
    pick.sort()

    texts = [abstracts[i] for i in pick]
    del abstracts

    vec = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
        min_df=10,
        max_df=0.3,
        max_features=60_000,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    vocab_all = vec.get_feature_names_out()

    # Drop docs with too few in-vocabulary tokens, then trim to exactly n_docs.
    keep = np.where(X.getnnz(axis=1) >= 5)[0]
    if keep.size > n_docs:
        keep = keep[rng.choice(keep.size, size=n_docs, replace=False)]
        keep.sort()
    X = X[keep]
    pick = pick[keep]

    svd = TruncatedSVD(n_components=LSA_DIM, random_state=seed)
    Z = svd.fit_transform(X).astype(np.float32)
    Z = normalize(Z).astype(np.float32)

    # Keyword space: top-N terms by document frequency within the sample.
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    kw_cols = np.argsort(df)[::-1][:N_KEYWORDS]
    kw_cols.sort()
    K = (X[:, kw_cols] > 0).astype(np.uint8).tocsr()
    vocab = vocab_all[kw_cols]

    # Foreground: a coherent KMeans cluster with healthy bg interleaving.
    km = MiniBatchKMeans(n_clusters=KMEANS_K, random_state=seed, n_init=5, batch_size=4096)
    labels = km.fit_predict(Z)
    sizes = np.bincount(labels, minlength=KMEANS_K)
    lo, hi = FG_SIZE_RANGE
    candidates = [c for c in range(KMEANS_K) if lo <= sizes[c] <= hi]
    if not candidates:  # fall back to closest-to-range
        candidates = [int(np.argmin(np.abs(sizes - (lo + hi) / 2)))]

    diagnostics = []
    for c in candidates:
        members = np.where(labels == c)[0]
        coherence = float(
            np.mean(Z[members] @ km.cluster_centers_[c] / np.linalg.norm(km.cluster_centers_[c]))
        )
        samp = members[rng.choice(members.size, size=min(300, members.size), replace=False)]
        nn_fg = NearestNeighbors(n_neighbors=11, algorithm="brute").fit(Z[members])
        r_ff = nn_fg.kneighbors(Z[samp])[0][:, -1]  # 10th member-neighbor (col 0 is self)
        nonmembers = np.where(labels != c)[0]
        nn_bg = NearestNeighbors(n_neighbors=10, algorithm="brute").fit(Z[nonmembers])
        d_bg, _ = nn_bg.kneighbors(Z[samp])
        interleave = float(np.median((d_bg <= r_ff[:, None]).sum(axis=1)))
        diagnostics.append(
            {"cluster": int(c), "size": int(sizes[c]), "coherence": round(coherence, 4),
             "median_bg_within_r10fg": interleave}
        )

    # Require interleaving (bg mass inside fg territory), then take most coherent.
    ok = [d for d in diagnostics if d["median_bg_within_r10fg"] >= 5]
    pool = ok if ok else diagnostics
    chosen = max(pool, key=lambda d: d["coherence"])
    fg_mask = labels == chosen["cluster"]

    # Top terms of the chosen cluster (mean tf-idf), an allowed aggregate.
    mean_tfidf = np.asarray(X[fg_mask].mean(axis=0)).ravel()
    top_terms = [str(vocab_all[i]) for i in np.argsort(mean_tfidf)[::-1][:25]]

    meta = {
        "source": str(WIKI_PARQUET.relative_to(REPO)),
        "seed": seed,
        "n_docs": int(Z.shape[0]),
        "lsa_dim": LSA_DIM,
        "n_keywords": N_KEYWORDS,
        "tfidf_vocab": int(len(vocab_all)),
        "svd_explained_variance_ratio_sum": round(float(svd.explained_variance_ratio_.sum()), 4),
        "kmeans_k": KMEANS_K,
        "fg_cluster": chosen,
        "fg_top_terms": top_terms,
        "cluster_candidates": diagnostics,
    }
    return Dataset(
        Z=Z, K=K, vocab=np.asarray(vocab, dtype=object),
        titles=titles[pick], page_ids=page_ids[pick].astype(np.int64),
        fg_mask=fg_mask, meta=meta,
    )


def load_dataset(rebuild: bool = False) -> Dataset:
    """Load the cached dataset, building it on first use."""
    npz = CACHE_DIR / "dataset.npz"
    knpz = CACHE_DIR / "keywords.npz"
    metaj = CACHE_DIR / "dataset_meta.json"
    if not rebuild and npz.exists() and knpz.exists() and metaj.exists():
        d = np.load(npz, allow_pickle=True)
        K = sp.load_npz(knpz).tocsr()
        meta = json.loads(metaj.read_text())
        return Dataset(
            Z=d["Z"], K=K, vocab=d["vocab"], titles=d["titles"],
            page_ids=d["page_ids"], fg_mask=d["fg_mask"], meta=meta,
        )
    ds = _build()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz, Z=ds.Z, vocab=ds.vocab, titles=ds.titles,
        page_ids=ds.page_ids, fg_mask=ds.fg_mask,
    )
    sp.save_npz(knpz, ds.K)
    metaj.write_text(json.dumps(ds.meta, indent=2))
    return ds


def main() -> None:
    ds = load_dataset()
    print(json.dumps({k: v for k, v in ds.meta.items() if k != "cluster_candidates"}, indent=2))
    print(f"fg={int(ds.fg_mask.sum())} bg={int((~ds.fg_mask).sum())} "
          f"Z={ds.Z.shape} K={ds.K.shape} nnz/doc={ds.K.getnnz() / ds.K.shape[0]:.1f}")


if __name__ == "__main__":
    main()
