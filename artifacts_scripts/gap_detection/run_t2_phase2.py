"""Track 2 Phase 2 runner: build the leakage-free T-corpus + frozen embeddings.

Inputs: t2_dates.npz (Phase 1), t2_choice.json (Phase 0 pre-commit).
Steps:
  1. pre-T eligible pool from the 400k store (creation <= T, abstract >= 200 chars)
  2. fit_t_corpus: 30k uniform sample (banked pre-T docs force-included),
     TF-IDF/LSA-128/top-2000-keywords/KMeans-40 fg selection — all <=T-fitted
  3. fg overlap sanity vs the banked fg (society/law/politics)
  4. scope: bg docs by 10th-nearest-fg distance, 0.25 quantile (T geometry)
  5. FROZEN-transform embeddings: arrivals (T, T+Delta], growth window (T-Delta, T]
Caches under artifacts/gap_detection_eval/cache/: t2_corpus.npz, t2_models.joblib,
t2_arrivals.npz, t2_growth.npz, t2_phase2_meta.json.

Context hygiene: reads abstracts internally; prints only counts/aggregates/terms.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import json
import time

import numpy as np
import scipy.sparse as sp

from artifacts_scripts.gap_detection.data import CACHE_DIR, MIN_ABSTRACT_CHARS, load_dataset
from artifacts_scripts.gap_detection.t_corpus import fg_overlap_sanity, fit_t_corpus
from artifacts_scripts.gap_detection.wiki_temporal import WIKI_PARQUET, year_to_days

_T0 = time.monotonic()
K_NN = 10
SCOPE_QUANTILE = 0.25


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    import pyarrow.parquet as pq

    choice = json.loads((CACHE_DIR / "t2_choice.json").read_text())
    ty, tm, _ = (int(x) for x in choice["T"].split("-"))
    t_days = year_to_days(ty, tm)
    # calendar-accurate window end (T + delta years)
    end_days = year_to_days(ty + choice["delta_years"], tm)
    log(f"T = {choice['T']} (day {t_days:.1f}), window end day {end_days:.1f}")

    d = np.load(CACHE_DIR / "t2_dates.npz")
    store_days = d["store_days"]

    tbl = pq.read_table(WIKI_PARQUET)
    page_ids = tbl.column("page_id").to_numpy().astype(np.int64)
    titles = np.asarray(tbl.column("title").to_pylist(), dtype=object)
    abstracts = tbl.column("abstract").to_pylist()
    del tbl
    lengths = np.fromiter((len(a or "") for a in abstracts), dtype=np.int64)
    elig_chars = lengths >= MIN_ABSTRACT_CHARS

    pre_mask = (store_days <= t_days) & elig_chars
    pool_rows = np.where(pre_mask)[0]
    log(f"pre-T eligible pool: {pool_rows.size:,} store rows")

    ds = load_dataset()
    sub_days = d["sub_days"]
    pos = {int(p): i for i, p in enumerate(page_ids)}
    banked_rows = np.array([pos[int(p)] for p in ds.page_ids], dtype=np.int64)
    part_a_rows = banked_rows[sub_days <= t_days]
    # part A must be a subset of the pool (banked docs pass the chars rule)
    a_in_pool = np.isin(part_a_rows, pool_rows)
    log(f"banked pre-T docs (part A): {part_a_rows.size:,} ({int(a_in_pool.sum()):,} in pool)")
    part_a_rows = part_a_rows[a_in_pool]

    tc = fit_t_corpus(
        texts=[abstracts[i] for i in pool_rows],
        days=store_days[pool_rows],
        t_days=t_days,
        titles=titles[pool_rows],
        page_ids=page_ids[pool_rows],
        store_rows=pool_rows,
        part_a_rows=part_a_rows,
        seed=42,
    )
    m = tc.meta
    log(f"T-corpus: n={m['n_docs']:,} vocab={m['tfidf_vocab']:,} "
        f"EVR={m['svd_evr_sum']} part_a_kept={int(np.isin(tc.store_rows, part_a_rows).sum()):,}")
    log(f"fg cluster: {m['fg_cluster']}")
    log(f"fg top terms: {', '.join(m['fg_top_terms'][:12])}")

    banked_fg_pids = ds.page_ids[ds.fg_mask]
    sanity = fg_overlap_sanity(tc, banked_fg_pids)
    log(f"fg overlap sanity (procedure pick): {json.dumps(sanity)}")

    # ---- FIELD-CONTINUITY RULE (deviation on record): the seeded procedure's
    # coherence-max pick must be checked against the banked field. If it is not
    # the max-banked-overlap cluster among SIZE-QUALIFIED candidates, the field
    # did not re-form under the procedure (the banked society/law field
    # fragments at T) and the fg is set to the max-overlap size-qualified
    # cluster instead. The interleave gate is plant-era fixture machinery
    # (needed bg mass inside fg for orphan detection) and takes no part here.
    present = np.isin(tc.page_ids, banked_fg_pids)
    lab = tc.kmeans.labels_
    kk = int(m["kmeans_k"])
    hist = np.bincount(lab[present], minlength=kk)
    sizes = np.bincount(lab, minlength=kk)
    lo_sz, hi_sz = 1_000, 3_000
    qualified = np.where((sizes >= lo_sz) & (sizes <= hi_sz))[0]
    best_overlap_cluster = int(qualified[np.argmax(hist[qualified])])
    log(f"max-overlap size-qualified cluster: {best_overlap_cluster} "
        f"(banked members {int(hist[best_overlap_cluster])}/{int(present.sum())}, "
        f"size {int(sizes[best_overlap_cluster])})")
    procedure_pick = int(m["kmeans_labels_chosen"])
    fg_rule = "procedure"
    if procedure_pick != best_overlap_cluster:
        fg_rule = "max_overlap_size_qualified"
        tc.fg_mask = lab == best_overlap_cluster
        tc.meta["kmeans_labels_chosen"] = best_overlap_cluster
        # distinctive terms of the final fg (log-odds over the T keyword space)
        from artifacts_scripts.gap_detection.polish import cluster_vocab

        Kd = np.asarray(tc.K.todense(), dtype=bool)
        cols = cluster_vocab(Kd, tc.fg_mask, n_top=15)
        m["fg_final_terms"] = [str(tc.vocab[j]) for j in cols]
        log(f"fg FINAL = cluster {best_overlap_cluster} (rule: {fg_rule}); "
            f"terms: {', '.join(m['fg_final_terms'][:12])}")
    sanity_final = fg_overlap_sanity(tc, banked_fg_pids)
    log(f"fg overlap sanity (final): {json.dumps(sanity_final)}")

    # ---- scope (T geometry, fg/bg of the T-corpus)
    from artifacts_scripts.gap_detection.detectors import knn_radii

    Zf, Zb = tc.Z[tc.fg_idx], tc.Z[tc.bg_idx]
    r_F10_bg = knn_radii(Zb, Zf, K_NN)
    thr = float(np.quantile(r_F10_bg, SCOPE_QUANTILE))
    scoped_idx = np.where(r_F10_bg <= thr)[0]
    log(f"scope: thr={thr:.4f} scoped_bg={scoped_idx.size:,} of {Zb.shape[0]:,}")

    # ---- frozen-transform embeddings: arrivals + growth window
    def embed_rows(rows: np.ndarray, label: str):
        Z_chunks, K_chunks, nnz_chunks = [], [], []
        for s in range(0, rows.size, 50_000):
            rr = rows[s : s + 50_000]
            Zc, Kc, nz = tc.transform_texts([abstracts[i] or "" for i in rr])
            Z_chunks.append(Zc)
            K_chunks.append(Kc)
            nnz_chunks.append(nz)
        Z = np.vstack(Z_chunks)
        K = sp.vstack(K_chunks).tocsr()
        nnz = np.concatenate(nnz_chunks)
        ok = nnz >= 5
        log(f"{label}: {rows.size:,} rows, eligible (nnz>=5) {int(ok.sum()):,}")
        return Z[ok], K[ok], rows[ok]

    arr_mask = (store_days > t_days) & (store_days <= end_days) & elig_chars
    arr_rows_all = np.where(arr_mask)[0]
    Z_arr, K_arr, arr_rows = embed_rows(arr_rows_all, "arrivals")

    # growth window = ALL store pages created in (T-2y, T] (corpus + non-corpus)
    grow_mask_full = (store_days > t_days - 2 * 365.2425) & (store_days <= t_days) & elig_chars
    grow_rows_all = np.where(grow_mask_full)[0]
    Z_grow, _K_grow, grow_rows = embed_rows(grow_rows_all, "growth-window (T-2y, T]")

    np.savez_compressed(
        CACHE_DIR / "t2_corpus.npz",
        Z=tc.Z, vocab=tc.vocab, titles=tc.titles, page_ids=tc.page_ids,
        store_rows=tc.store_rows, days=tc.days, fg_mask=tc.fg_mask,
        kmeans_labels=tc.kmeans.labels_, scoped_idx=scoped_idx, scope_thr=thr,
    )
    sp.save_npz(CACHE_DIR / "t2_corpus_K.npz", tc.K)
    np.savez_compressed(CACHE_DIR / "t2_arrivals.npz",
                        Z=Z_arr, store_rows=arr_rows, days=store_days[arr_rows],
                        titles=titles[arr_rows], page_ids=page_ids[arr_rows])
    sp.save_npz(CACHE_DIR / "t2_arrivals_K.npz", K_arr)
    np.savez_compressed(CACHE_DIR / "t2_growth.npz",
                        Z=Z_grow, store_rows=grow_rows, days=store_days[grow_rows])
    import joblib

    joblib.dump({"vectorizer": tc.vectorizer, "svd": tc.svd, "kmeans": tc.kmeans,
                 "kw_cols": tc.meta["kw_cols"]},
                CACHE_DIR / "t2_models.joblib", compress=3)
    meta = {
        "choice": choice, "t_days": float(t_days), "end_days": float(end_days),
        "n_pool": int(pool_rows.size), "n_part_a": int(part_a_rows.size),
        "corpus": {k: (v if not isinstance(v, np.ndarray) else None)
                   for k, v in m.items() if k != "kw_cols"},
        "fg_rule": fg_rule,
        "fg_procedure_pick": procedure_pick,
        "fg_overlap_sanity_procedure": sanity,
        "fg_overlap_sanity_final": sanity_final,
        "banked_fragmentation_top5": [
            {"cluster": int(c), "n_banked": int(hist[c]), "size": int(sizes[c])}
            for c in np.argsort(hist)[::-1][:5] if hist[c] > 0
        ],
        "max_overlap_cluster": {"cluster": best_overlap_cluster,
                                "n_banked": int(hist[best_overlap_cluster]),
                                "size": int(sizes[best_overlap_cluster])},
        "scope": {"thr": thr, "n_scoped": int(scoped_idx.size), "n_bg": int(Zb.shape[0]),
                  "quantile": SCOPE_QUANTILE},
        "arrivals": {"n_raw": int(arr_rows_all.size), "n_eligible": int(arr_rows.size)},
        "growth_window": {"n_raw": int(grow_rows_all.size), "n_eligible": int(grow_rows.size),
                          "window_years": 2.0},
        "approximations_on_record": [
            "2026 text for pre-T articles (drift): abstracts are the 2025-12-29 cirrus "
            "revisions, not the <=T revisions",
            "deletion survivorship: only pages surviving to the dump exist in the store "
            "(both corpus and arrivals)",
            "undated/proxy dates: creation = isotonic id->date proxy with exact API "
            "overrides on the calibration sample; near-boundary misclassification is "
            "symmetric (validation p90 reported in t2_calibration_report.json)",
            "store ceiling 2004-12-31 truncates the arrival window by 1 day",
        ],
    }
    (CACHE_DIR / "t2_phase2_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    log("wrote t2_corpus(.npz/_K.npz), t2_arrivals(.npz/_K.npz), t2_growth.npz, "
        "t2_models.joblib, t2_phase2_meta.json")


if __name__ == "__main__":
    main()
