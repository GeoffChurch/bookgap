"""Compare K-ball scoring options against FF median_logit final ordering.

For each existing K-ball candidate gap, compute four candidate scoring
functions:

  - A (abs_density): mean kernel mass from gap centroid to its members
  - B (llr_null):    log(kernel_mass) - null_log_mean[k]  (LLR vs corpus null)
  - C (llr_kb):      K_G / (K_G + K_B) via local-CSLS-balanced K_B
                     (the FF posterior at K-ball-ball-size)
  - D (size_x_llr):  ball_size × LLR(seed)

For each scoring, rank K-ball gaps. Compute Spearman correlation
between that K-ball ranking and the FF median_logit ranking (on
the survivor subset, which has both numbers).

Also times per-candidate scoring cost.

Usage::

    uv run python -u -m artifacts_scripts.gap_kball_scoring_compare \\
        --scope 17/1702/T11303 \\
        --kball-csv artifacts/kball_gap_probe/17-1702-T11303__pca__drop0.csv \\
        --refined-csv artifacts/gap_kball_llr_refine/T11303_n1000__llr_refined.csv \\
        --out cache/gap_kball_scoring_compare/T11303.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from artifacts_scripts.gap_static_seed_diagnostic import (
    K_GRID,
    K_QUERY,
    kernel_mass_matrix,
    setup,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True)
    p.add_argument("--kball-csv", required=True, type=Path)
    p.add_argument("--refined-csv", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--min-kball-sizes",
        default="0,5,10,20,30",
        help="Comma-separated K-ball gap_size thresholds; "
             "report Spearman at each threshold (default 0,5,10,20,30).",
    )
    p.add_argument(
        "--scores-csv",
        type=Path,
        default=None,
        help="Optional path to dump per-gap A/B/C/D scores for later re-analysis.",
    )
    args = p.parse_args()
    min_sizes = [int(x) for x in args.min_kball_sizes.split(",")]

    print(f"[load] reading K-ball CSV {args.kball_csv}...", flush=True)
    kball_rows = []
    with open(args.kball_csv) as f:
        for r in csv.DictReader(f):
            kball_rows.append({
                "gap_idx": int(r["gap_idx"]),
                "members": [int(x) for x in json.loads(r["article_indices_json"])],
                "size": int(r["gap_size"]),
            })
    print(f"  {len(kball_rows)} K-ball candidates", flush=True)

    print(f"[load] reading refined CSV {args.refined_csv}...", flush=True)
    refined_by_id = {}
    with open(args.refined_csv) as f:
        for r in csv.DictReader(f):
            refined_by_id[int(r["kball_gap_id"])] = {
                "median_logit": float(r["median_logit"]),
                "survivor": int(r["survivor"]),
            }
    n_survivors = sum(1 for r in refined_by_id.values() if r["survivor"] == 1)
    print(f"  {len(refined_by_id)} refined entries, {n_survivors} survivors", flush=True)

    print(f"[setup] loading scope {args.scope}...", flush=True)
    state = setup(args.scope, index_type="flat")  # exact for diagnostic
    emb = state["emb"]
    sigma = state["sigma"]
    article_index = state["article_index"]
    article_global_idx = state["article_global_idx"]
    book_global_idx = state["book_global_idx"]
    n_total = state["sigma"].shape[0]

    from bookgap.index import build_index, FlatL2
    book_emb = np.ascontiguousarray(emb[book_global_idx])
    book_index = build_index(FlatL2(), book_emb)

    print(f"[null] computing corpus null at K_GRID...", flush=True)
    art_idx = state["article_global_idx"]
    M_null = kernel_mass_matrix(
        D_art_NN=state["D_art_NN_all"][art_idx],
        I_art_NN_global=state["I_art_NN_all_global"][art_idx],
        sigma=sigma,
        k_grid=K_GRID,
        seed_global_idx=art_idx,
    )
    null_log_mean = np.log(np.maximum(M_null, 1e-30)).mean(axis=0)

    def _null_at_k(k: int) -> float:
        """Interpolate null_log_mean to arbitrary k by nearest K_GRID value."""
        nearest_idx = int(np.argmin(np.abs(np.array(K_GRID) - k)))
        return float(null_log_mean[nearest_idx])

    print(f"[score] computing 4 scoring functions for {len(kball_rows)} gaps...",
          flush=True)
    times = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    scores_A: list[float] = []  # abs_density
    scores_B: list[float] = []  # llr_null
    scores_C: list[float] = []  # llr_kb (FF posterior at ball size)
    scores_D: list[float] = []  # size × llr

    for i, kg in enumerate(kball_rows):
        members = np.asarray(kg["members"], dtype=np.int64)
        if len(members) < 2:
            scores_A.append(0.0); scores_B.append(0.0)
            scores_C.append(0.0); scores_D.append(0.0)
            continue
        member_emb = emb[members]
        centroid = member_emb.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-12:
            scores_A.append(0.0); scores_B.append(0.0)
            scores_C.append(0.0); scores_D.append(0.0)
            continue
        centroid_unit = (centroid / norm).astype(np.float32)
        sigma_seed = float(sigma[members].mean())
        sigma_members = sigma[members]
        d_sq = np.sum((member_emb - centroid_unit) ** 2, axis=1)
        w_centroid_to_member = np.exp(-d_sq / (sigma_seed * sigma_members))

        # --- A: abs_density: mean kernel weight from centroid to members ---
        t0 = time.perf_counter()
        score_A = float(w_centroid_to_member.mean())
        times["A"] += time.perf_counter() - t0
        scores_A.append(score_A)

        # --- B: llr_null: log(kernel_mass) - null at ball size ---
        t0 = time.perf_counter()
        kernel_mass = float(w_centroid_to_member.sum())
        score_B = float(np.log(max(kernel_mass, 1e-30))) - _null_at_k(len(members))
        times["B"] += time.perf_counter() - t0
        scores_B.append(score_B)

        # --- D: size_x_llr ---
        scores_D.append(score_B * len(members))
        # times["D"] same as B; negligible

        # --- C: llr_kb (FF posterior at ball size) ---
        # Need per-member K_G and K_B via local-CSLS background.
        t0 = time.perf_counter()
        # Per-member kNN
        full_index = state["full_index"]
        member_emb_c = np.ascontiguousarray(member_emb)
        D_NN, I_NN_full = full_index.search(member_emb_c, k=K_QUERY + 1)
        D_NN = D_NN[:, 1:]
        I_NN_full = I_NN_full[:, 1:]
        # 1-NN book per member
        _, I_book = book_index.search(member_emb_c, k=1)
        nearest_book = book_global_idx[I_book[:, 0]]
        unique_books = np.unique(nearest_book)
        if len(unique_books) > 0:
            book_emb_q = np.ascontiguousarray(emb[unique_books])
            _, I_bart_local = article_index.search(book_emb_q, k=K_QUERY)
            book_to_arts = {
                int(b): article_global_idx[I_bart_local[r2]]
                for r2, b in enumerate(unique_books)
            }
        else:
            book_to_arts = {}
        in_G_hard = np.zeros(n_total, dtype=bool)
        in_G_hard[members] = True
        bcount: Counter = Counter()
        for b in nearest_book:
            bcount[int(b)] += 1
        in_B = np.zeros(n_total, dtype=bool)
        for b, f_b in bcount.items():
            if b < 0:
                continue
            in_B[b] = True
            arts = book_to_arts.get(b, np.array([], dtype=np.int64))
            taken = 0
            for a in arts:
                if taken >= f_b:
                    break
                a_int = int(a)
                if not in_G_hard[a_int]:
                    in_B[a_int] = True
                    taken += 1
        sigma_seed_arr = sigma[members][:, None]
        sigma_n = sigma[I_NN_full]
        W = np.exp(-D_NN / (sigma_seed_arr * sigma_n))
        self_mask = I_NN_full == members[:, None]
        G_mask = in_G_hard[I_NN_full] & ~self_mask
        B_mask = in_B[I_NN_full]
        K_G = (W * G_mask).sum(axis=1)
        K_B = (W * B_mask).sum(axis=1)
        denom = K_G + K_B + 1e-30
        posts = K_G / denom
        score_C = float(np.median(posts))  # median posterior at K-ball ball size
        times["C"] += time.perf_counter() - t0
        scores_C.append(score_C)

        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{len(kball_rows)} "
                  f"(A={times['A']:.2f}s B={times['B']:.2f}s "
                  f"C={times['C']:.2f}s)", flush=True)

    # --- Optionally dump per-gap scores for re-analysis ---
    if args.scores_csv is not None:
        args.scores_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.scores_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "kball_gap_id", "kball_size", "survivor", "ff_median_logit",
                "A_abs_density", "B_llr_null", "C_llr_kb", "D_size_x_llr",
            ])
            for i, kg in enumerate(kball_rows):
                ref = refined_by_id.get(kg["gap_idx"], {})
                w.writerow([
                    kg["gap_idx"], kg["size"],
                    ref.get("survivor", 0),
                    f"{ref.get('median_logit', float('nan')):.6f}"
                        if ref.get("median_logit") is not None else "",
                    f"{scores_A[i]:.6e}",
                    f"{scores_B[i]:.6e}",
                    f"{scores_C[i]:.6e}",
                    f"{scores_D[i]:.6e}",
                ])
        print(f"[dump] per-gap scores written to {args.scores_csv}", flush=True)

    # --- Spearman correlations (on survivors), at multiple min_kball_size cuts ---
    print(f"[rank] computing Spearman vs FF median_logit at multiple size cuts...",
          flush=True)
    from scipy.stats import spearmanr
    out_table: list[dict] = []
    for min_size in min_sizes:
        survivor_idx = [
            i for i, kg in enumerate(kball_rows)
            if refined_by_id.get(kg["gap_idx"], {}).get("survivor", 0) == 1
            and kg["size"] >= min_size
        ]
        if len(survivor_idx) < 3:
            print(f"  min_kball_size={min_size}: n={len(survivor_idx)} "
                  f"survivors (too few)", flush=True)
            continue
        ff_med = [refined_by_id[kball_rows[i]["gap_idx"]]["median_logit"]
                  for i in survivor_idx]
        A_sur = [scores_A[i] for i in survivor_idx]
        B_sur = [scores_B[i] for i in survivor_idx]
        C_sur = [scores_C[i] for i in survivor_idx]
        D_sur = [scores_D[i] for i in survivor_idx]
        rho_A, _ = spearmanr(A_sur, ff_med)
        rho_B, _ = spearmanr(B_sur, ff_med)
        rho_C, _ = spearmanr(C_sur, ff_med)
        rho_D, _ = spearmanr(D_sur, ff_med)
        kball_idx_sur = [-kball_rows[i]["gap_idx"] for i in survivor_idx]
        rho_kball, _ = spearmanr(kball_idx_sur, ff_med)
        print(f"  min_kball_size={min_size:>3}  n={len(survivor_idx):>4}  "
              f"K-ball:{rho_kball:+.3f}  A:{rho_A:+.3f}  B:{rho_B:+.3f}  "
              f"C:{rho_C:+.3f}  D:{rho_D:+.3f}",
              flush=True)
        out_table.append({
            "min_kball_size": min_size,
            "n_survivors": len(survivor_idx),
            "rho_kball": rho_kball,
            "rho_A": rho_A,
            "rho_B": rho_B,
            "rho_C": rho_C,
            "rho_D": rho_D,
        })

    print()
    print(f"  per-scoring total time over {len(kball_rows)} candidates:")
    print(f"    A: {times['A']:.2f}s  ({times['A']*1000/len(kball_rows):.2f}ms / candidate)")
    print(f"    B: {times['B']:.2f}s  ({times['B']*1000/len(kball_rows):.2f}ms / candidate)")
    print(f"    C: {times['C']:.2f}s  ({times['C']*1000/len(kball_rows):.2f}ms / candidate)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "min_kball_size", "n_survivors",
            "rho_kball_size_based",
            "rho_A_abs_density",
            "rho_B_llr_null",
            "rho_C_llr_kb",
            "rho_D_size_x_llr",
        ])
        for row in out_table:
            w.writerow([
                row["min_kball_size"], row["n_survivors"],
                f"{row['rho_kball']:+.4f}",
                f"{row['rho_A']:+.4f}",
                f"{row['rho_B']:+.4f}",
                f"{row['rho_C']:+.4f}",
                f"{row['rho_D']:+.4f}",
            ])
    print(f"[done] csv written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
