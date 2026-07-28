"""Track 2 pre-check support: embed ALL 400k store abstracts via the BANKED
2026-fitted transform (rebuilt + identity-asserted), for the Phase-0 gate's
in-scope arrival counts ONLY. (Phase 2/3 use a freshly fit <=T transform;
this banked embedding is never used as a predictor or outcome.)

Writes artifacts/gap_detection_eval/cache/t2_store_embed.npz:
  Z (400k, 128) float32 L2-normalized; eligible (400k,) bool
  (abstract >= 200 chars AND >= 5 in-vocab terms — the corpus eligibility rule).

Context hygiene: reads raw abstracts internally; emits matrices/counts only.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import time

import numpy as np

from artifacts_scripts.gap_detection.data import CACHE_DIR, MIN_ABSTRACT_CHARS, load_dataset

_T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    import pyarrow.parquet as pq
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    from artifacts_scripts.gap_detection.data import N_DOCS, SEED, WIKI_PARQUET

    ds = load_dataset()
    # ---- replay data._build's vectorizer fit exactly (same as polish.rebuild_tfidf,
    # but keeping the fitted vectorizer for store-wide transform)
    rng = np.random.default_rng(SEED)
    tbl = pq.read_table(WIKI_PARQUET)
    page_ids = tbl.column("page_id").to_numpy()
    abstracts = tbl.column("abstract").to_pylist()
    del tbl
    lengths = np.fromiter((len(a or "") for a in abstracts), dtype=np.int64)
    eligible_chars = lengths >= MIN_ABSTRACT_CHARS
    pick = rng.choice(np.where(eligible_chars)[0],
                      size=min(N_DOCS + 2_000, int(eligible_chars.sum())), replace=False)
    pick.sort()
    texts = [abstracts[i] for i in pick]
    vec = TfidfVectorizer(
        stop_words="english", lowercase=True, token_pattern=r"(?u)\b[a-z][a-z]+\b",
        min_df=10, max_df=0.3, max_features=60_000, sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    keep = np.where(X.getnnz(axis=1) >= 5)[0]
    if keep.size > N_DOCS:
        keep = keep[rng.choice(keep.size, size=N_DOCS, replace=False)]
        keep.sort()
    X = X[keep]
    pick = pick[keep]
    if not (page_ids[pick] == ds.page_ids).all():
        raise RuntimeError("banked transform rebuild: page_id mismatch")
    log(f"vectorizer rebuilt and identity-checked on {X.shape[0]} docs")

    svd = TruncatedSVD(n_components=128, random_state=SEED)
    Z_fit = normalize(svd.fit_transform(X).astype(np.float32)).astype(np.float32)
    if not np.allclose(Z_fit, ds.Z, atol=1e-4):
        raise RuntimeError("banked transform rebuild: LSA mismatch with cached Z")
    log(f"SVD rebuilt; max |Z - banked Z| = {np.abs(Z_fit - ds.Z).max():.2e}")

    # ---- transform the FULL store in chunks
    n = len(abstracts)
    Z = np.zeros((n, 128), dtype=np.float32)
    nnz = np.zeros(n, dtype=np.int64)
    chunk = 50_000
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        Xc = vec.transform([a or "" for a in abstracts[sl]])
        nnz[sl] = Xc.getnnz(axis=1)
        Zc = svd.transform(Xc).astype(np.float32)
        norms = np.linalg.norm(Zc, axis=1, keepdims=True)
        Z[sl] = Zc / np.maximum(norms, 1e-9)
        log(f"transformed {sl.stop}/{n}")
    eligible = eligible_chars & (nnz >= 5)
    # identity: store rows of the 30k subsample must match cached Z
    pos = {int(p): i for i, p in enumerate(page_ids)}
    rows = np.array([pos[int(p)] for p in ds.page_ids])
    err = float(np.abs(Z[rows] - ds.Z).max())
    if err > 5e-3:
        raise RuntimeError(f"store transform identity failed: max err {err}")
    log(f"store-row identity vs cached subsample Z: max err {err:.2e}")
    np.savez_compressed(CACHE_DIR / "t2_store_embed.npz", Z=Z, eligible=eligible)
    log(f"wrote t2_store_embed.npz (eligible: {int(eligible.sum()):,}/{n:,})")


if __name__ == "__main__":
    main()
