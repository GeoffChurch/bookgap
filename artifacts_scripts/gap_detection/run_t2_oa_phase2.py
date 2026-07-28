"""Track 2 OA phase — Phase 2: leakage-free T-corpus + frozen-transform embeddings.

Mirrors run_t2_phase2.py (wiki), data layer = OpenAlex (oa_corpus). Reuses
`t_corpus.fit_t_corpus` UNCHANGED (every fitted object on ≤T docs only; arrivals
via the frozen transform). Differences from wiki:
  * texts come from decoded abstract_inverted_index (not a parquet column);
  * publication_date is EXACT (no id→date proxy);
  * fg = the DISCOVERED KMeans cluster (no banked label, no continuity rule):
    the most-coherent size-qualified cluster, reported by its top terms;
  * missingness model: works without an abstract (>=200 chars) are DROPPED from
    the corpus + the embedding (documented), exactly as the wiki char rule.

Outputs (cache/): t2_oa_corpus(.npz/_K.npz), t2_oa_models.joblib,
t2_oa_arrivals(.npz/_K.npz), t2_oa_growth.npz, t2_oa_phase2_meta.json, and the
dated-graph COO (t2_oa_graph.npz) for Phase 3's P3a + voltage-on-graph.

Context hygiene: abstracts decoded internally; prints counts/aggregates/terms.
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

from artifacts_scripts.gap_detection.oa_corpus import (
    CACHE_DIR,
    UNIVERSE_TOPIC,
    build_dated_graph,
    decode_for_rows,
    load_works_table,
    year_to_days,
)
from artifacts_scripts.gap_detection.t_corpus import fit_t_corpus

_T0 = time.monotonic()
K_NN = 10
SCOPE_QUANTILE = 0.25
MIN_ABSTRACT_CHARS = 200
PRE_FIT_CAP = 80_000  # subsample the pre-T pool to this for the kNN/spectral pipeline


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    choice = json.loads((CACHE_DIR / "t2_oa_choice.json").read_text())
    topics = tuple(choice["topics"])
    sub_key = "primary_topic.id:" + "|".join(topics)
    ty = int(choice["T"].split("-")[0])
    t_days = year_to_days(ty)
    end_days = year_to_days(ty + choice["delta_years"])
    log(f"T = {choice['T']} (day {t_days:.0f}), window end {choice['window']} "
        f"(day {end_days:.0f}); universe={choice['universe']}")

    tbl = load_works_table(sub_key, log=log)
    wid, days = tbl["wid"], tbl["days"]
    abs_chars = tbl["abstract_chars"]
    english_ok = tbl["english_ok"]
    has_date = np.isfinite(days)
    # eligibility = abstract >=200ch + English (drop the ~5% non-English language island)
    elig_chars = (abs_chars >= MIN_ABSTRACT_CHARS) & english_ok

    # ----- missingness model (documented): keep only dated + abstracted(>=200ch) + English
    pre_mask = has_date & (days <= t_days) & elig_chars
    pool_rows = np.where(pre_mask)[0]
    n_pre_dated = int((has_date & (days <= t_days)).sum())
    n_pre_abstracted = int(pre_mask.sum())
    n_pre_nonen = int((has_date & (days <= t_days) & (abs_chars >= MIN_ABSTRACT_CHARS)
                       & ~english_ok).sum())
    log(f"pre-T pool: dated {n_pre_dated:,} -> with English abstract>=200ch "
        f"{n_pre_abstracted:,} (dropped: no/short abstract + {n_pre_nonen:,} non-English)")

    # subsample the pre-T pool for the fit if huge (seeded; documented)
    rng = np.random.default_rng(42)
    fit_rows = pool_rows
    subsampled = False
    if pool_rows.size > PRE_FIT_CAP:
        fit_rows = np.sort(rng.choice(pool_rows, size=PRE_FIT_CAP, replace=False))
        subsampled = True
    log(f"T-corpus fit rows: {fit_rows.size:,}{' (subsampled)' if subsampled else ''}")

    texts = decode_for_rows(sub_key, fit_rows, wid)

    # fit_t_corpus expects a pool; n_docs caps the corpus. No banked part-A here
    # (fg is discovered, not matched), so part_a_rows is empty.
    n_docs = min(fit_rows.size, max(PRE_FIT_CAP, 30_000))
    tc = fit_t_corpus(
        texts=texts,
        days=days[fit_rows],
        t_days=t_days,
        titles=wid[fit_rows],                  # titles slot carries W-ids (hygiene: no text)
        page_ids=fit_rows.astype(np.int64),    # "page_ids" = universe row indices
        store_rows=fit_rows.astype(np.int64),
        part_a_rows=np.array([], dtype=np.int64),
        seed=42,
        n_docs=n_docs,
        min_interleave=0.0,   # OA fg is the discovered cluster; no plant-era interleave gate
    )
    m = tc.meta
    log(f"T-corpus: n={m['n_docs']:,} vocab={m['tfidf_vocab']:,} EVR={m['svd_evr_sum']}")
    log(f"fg cluster (DISCOVERED): {m['fg_cluster']}")
    log(f"fg top terms: {', '.join(m['fg_top_terms'][:15])}")

    fg_label = int(m["kmeans_labels_chosen"])

    # ----- scope (T geometry: fg/bg of the T-corpus)
    from artifacts_scripts.gap_detection.detectors import knn_radii

    Zf, Zb = tc.Z[tc.fg_idx], tc.Z[tc.bg_idx]
    r_F10_bg = knn_radii(Zb, Zf, K_NN)
    thr = float(np.quantile(r_F10_bg, SCOPE_QUANTILE))
    scoped_idx = np.where(r_F10_bg <= thr)[0]
    log(f"scope: thr={thr:.4f} scoped_bg={scoped_idx.size:,} of {Zb.shape[0]:,}")

    # ----- frozen-transform embeddings: arrivals (T, T+Δ] + growth window (T-Δ, T]
    def embed_rows(rows: np.ndarray, label: str):
        Z_chunks, K_chunks, nnz_chunks = [], [], []
        for s in range(0, rows.size, 50_000):
            rr = rows[s : s + 50_000]
            txt = decode_for_rows(sub_key, rr, wid)
            Zc, Kc, nz = tc.transform_texts(txt)
            Z_chunks.append(Zc)
            K_chunks.append(Kc)
            nnz_chunks.append(nz)
        Z = np.vstack(Z_chunks) if Z_chunks else np.zeros((0, tc.Z.shape[1]), np.float32)
        K = sp.vstack(K_chunks).tocsr() if K_chunks else sp.csr_matrix((0, tc.K.shape[1]))
        nnz = np.concatenate(nnz_chunks) if nnz_chunks else np.zeros(0, np.int64)
        ok = nnz >= 5
        log(f"{label}: {rows.size:,} rows, eligible (nnz>=5) {int(ok.sum()):,}")
        return Z[ok], K[ok], rows[ok]

    arr_mask = has_date & (days > t_days) & (days <= end_days) & elig_chars
    arr_rows_all = np.where(arr_mask)[0]
    Z_arr, K_arr, arr_rows = embed_rows(arr_rows_all, "arrivals (T, T+Δ]")

    grow_mask = has_date & (days > year_to_days(ty - choice["delta_years"])) \
        & (days <= t_days) & elig_chars
    grow_rows_all = np.where(grow_mask)[0]
    Z_grow, _Kg, grow_rows = embed_rows(grow_rows_all, "growth-window (T-Δ, T]")

    # ----- dated citation graph (COO) for Phase 3
    g = build_dated_graph(sub_key, tbl["row_of"], wid, days, log=log)

    # ----- persist
    np.savez_compressed(
        CACHE_DIR / "t2_oa_corpus.npz",
        Z=tc.Z, vocab=tc.vocab, wids=tc.titles, store_rows=tc.store_rows,
        days=tc.days, fg_mask=tc.fg_mask, kmeans_labels=tc.kmeans.labels_,
        scoped_idx=scoped_idx, scope_thr=thr,
    )
    sp.save_npz(CACHE_DIR / "t2_oa_corpus_K.npz", tc.K)
    np.savez_compressed(
        CACHE_DIR / "t2_oa_arrivals.npz",
        Z=Z_arr, store_rows=arr_rows, days=days[arr_rows], wids=wid[arr_rows],
    )
    sp.save_npz(CACHE_DIR / "t2_oa_arrivals_K.npz", K_arr)
    np.savez_compressed(
        CACHE_DIR / "t2_oa_growth.npz",
        Z=Z_grow, store_rows=grow_rows, days=days[grow_rows],
    )
    np.savez_compressed(
        CACHE_DIR / "t2_oa_graph.npz",
        src=g["src"], dst=g["dst"], citing_days=g["citing_days"], n_works=g["n_works"],
        universe_days=days, universe_has_date=has_date,
    )
    import joblib

    joblib.dump({"vectorizer": tc.vectorizer, "svd": tc.svd, "kmeans": tc.kmeans,
                 "kw_cols": tc.meta["kw_cols"]},
                CACHE_DIR / "t2_oa_models.joblib", compress=3)

    from artifacts_scripts.gap_detection.oa_corpus import graph_stats

    dated_T = has_date & (days <= t_days)
    dated_E = has_date & (days <= end_days)
    meta = {
        "choice": choice, "t_days": float(t_days), "end_days": float(end_days),
        "universe": choice["universe"], "topics": list(topics),
        "n_works_universe": int(wid.size), "n_pre_dated": n_pre_dated,
        "n_pre_abstracted": n_pre_abstracted, "n_pre_nonenglish_dropped": n_pre_nonen,
        "subsampled_prefit": subsampled,
        "pre_fit_cap": PRE_FIT_CAP,
        "corpus": {k: (v if not isinstance(v, np.ndarray) else None)
                   for k, v in m.items() if k != "kw_cols"},
        "fg_rule": "discovered_kmeans_coherence_max_size_qualified",
        "fg_label": fg_label, "fg_top_terms": m["fg_top_terms"],
        "scope": {"thr": thr, "n_scoped": int(scoped_idx.size),
                  "n_bg": int(Zb.shape[0]), "quantile": SCOPE_QUANTILE},
        "arrivals": {"n_raw": int(arr_rows_all.size), "n_eligible": int(arr_rows.size)},
        "growth_window": {"n_raw": int(grow_rows_all.size),
                          "n_eligible": int(grow_rows.size),
                          "window_years": float(choice["delta_years"])},
        "graph_T": graph_stats(g["at"](t_days), dated_T),
        "graph_T_plus_delta": graph_stats(g["at"](end_days), dated_E),
        "n_edges_intra_universe": int(g["src"].size),
        "approximations_on_record": [
            "abstract coverage < 100%: works without a decodable >=200ch abstract are "
            "dropped from the corpus AND the arrival/growth embedding (missingness model: "
            "drop; coverage reported in the gate table) — a no-abstract arrival cannot fill "
            "a neighborhood, so the fill counts are over abstracted arrivals only",
            f"language filter: {n_pre_nonen:,} pre-T non-English abstracts (~5%, mostly "
            "French) dropped — with stop_words=english their function words form a "
            "degenerate perfectly-coherent LANGUAGE cluster the coherence-max fg selector "
            "grabs; English-only keeps fg a genuine topical NLP cluster (is_english heuristic)",
            "pre-T pool subsampled to the fit cap for the kNN/spectral pipeline when large "
            "(seeded uniform; documented)" if subsampled else "no pre-T subsampling needed",
            "citation graph is intra-universe only (edges to works outside T10181 are not "
            "modeled); dated by the citing work's publication_date (papers cite earlier work "
            "-> exact T-graph, no future leakage)",
        ],
    }
    (CACHE_DIR / "t2_oa_phase2_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    log("wrote t2_oa_corpus(.npz/_K.npz), t2_oa_arrivals(.npz/_K.npz), t2_oa_growth.npz, "
        "t2_oa_graph.npz, t2_oa_models.joblib, t2_oa_phase2_meta.json")


if __name__ == "__main__":
    main()
