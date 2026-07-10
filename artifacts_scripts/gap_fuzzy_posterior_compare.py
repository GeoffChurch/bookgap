"""Compare hard vs fuzzy per-member posteriors for refined gaps.

For each survivor in a refined CSV, recompute K_G and K_B under three
gap-membership flavors and report the resulting (mean, median, min)
per-member logits:

  - HARD: in_G ∈ {0, 1} (current default, the "boolean" gap)
  - FUZZY_CENTROID: μ_G(j) = Gaussian kernel from gap centroid to j
                    (no cross-gap normalization)
  - FUZZY_JOINTNORM: μ_G(j) = (centroid-kernel weights) joint-normalized
                     so each article's memberships sum to 1 across gaps

K_B stays as the local-CSLS-balanced hard background in all three
flavors. (See docstring discussion for K_B fuzzification options.)

Usage::

    uv run python -u -m artifacts_scripts.gap_fuzzy_posterior_compare \\
        --scope 17/1702/T11303 \\
        --refined-csv artifacts/gap_kball_llr_refine/T11303_n1000__llr_refined.csv \\
        --out cache/gap_fuzzy_posterior_compare/T11303.csv \\
        --top-n 25
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from artifacts_scripts.gap_static_seed_diagnostic import K_QUERY, setup


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True)
    p.add_argument("--refined-csv", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--top-n", type=int, default=25,
                   help="Compute fuzzy stats for the top-N survivors by "
                        "median_logit (skip the rest to save compute)")
    args = p.parse_args()

    print(f"[load] reading {args.refined_csv}...", flush=True)
    all_rows = []
    with open(args.refined_csv) as f:
        for r in csv.DictReader(f):
            all_rows.append(r)
    survivors = [r for r in all_rows if r["survivor"] == "1"]
    survivors.sort(key=lambda r: -float(r["median_logit"]))
    targets = survivors[: args.top_n]
    print(f"  {len(all_rows)} rows, {len(survivors)} survivors, "
          f"analyzing top {len(targets)} by median_logit", flush=True)

    print(f"[setup] loading scope {args.scope}...", flush=True)
    state = setup(args.scope)
    print(f"  N_total={state['sigma'].shape[0]}", flush=True)

    from bookgap.index import build_index, FlatL2
    book_emb = np.ascontiguousarray(state["emb"][state["book_global_idx"]])
    book_index = build_index(FlatL2(), book_emb)
    article_index = state["article_index"]
    article_global_idx = state["article_global_idx"]
    book_global_idx = state["book_global_idx"]
    emb = state["emb"]
    sigma = state["sigma"]
    full_index = state["full_index"]
    n_total = state["sigma"].shape[0]

    # --- Build fuzzy memberships for all targets ---
    # For each target gap, compute centroid + kernel weights from
    # centroid to each member. These are the per-member "gap membership"
    # signals; FUZZY_CENTROID uses them raw, FUZZY_JOINTNORM normalizes
    # across gaps so each article's memberships sum to 1.
    print(f"[fuzzy] computing centroid-kernel memberships for top-{len(targets)} gaps...",
          flush=True)
    # For each target, list of (member_global_idx, raw_centroid_kernel_weight)
    target_member_weights: list[list[tuple[int, float]]] = []
    for r in targets:
        members = np.asarray(
            [int(x) for x in json.loads(r["article_indices_json"])], dtype=np.int64,
        )
        member_emb = emb[members]
        centroid = member_emb.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-12:
            target_member_weights.append([(int(m), 1.0) for m in members])
            continue
        centroid_unit = (centroid / norm).astype(np.float32)
        # Distances from centroid to each member
        d_centroid = np.linalg.norm(member_emb - centroid_unit, axis=1) ** 2
        # Use the mean of members' sigma as the centroid sigma.
        sigma_centroid = float(sigma[members].mean())
        sigma_members = sigma[members]
        w = np.exp(-d_centroid / (sigma_centroid * sigma_members))
        target_member_weights.append([
            (int(m), float(wi)) for m, wi in zip(members, w)
        ])

    # Build a sparse (n_total × n_targets) matrix of fuzzy memberships
    # for FUZZY_JOINTNORM.
    print(f"[fuzzy] joint-normalizing membership matrix...", flush=True)
    rows_idx, cols_idx, vals = [], [], []
    for g_col, mw in enumerate(target_member_weights):
        for m, w in mw:
            rows_idx.append(m)
            cols_idx.append(g_col)
            vals.append(w)
    M = csr_matrix(
        (vals, (rows_idx, cols_idx)),
        shape=(n_total, len(targets)),
    )
    row_sums = np.asarray(M.sum(axis=1)).flatten()
    from scipy.sparse import diags
    inv = np.where(row_sums > 0, 1.0 / np.where(row_sums > 0, row_sums, 1.0), 0.0)
    M_norm = (diags(inv) @ M).tocsr()
    # Convert to dense lookups (n_total × n_targets is small for top-N).
    M_dense_centroid = np.asarray(M.todense())
    M_dense_jointnorm = np.asarray(M_norm.todense())

    # --- Per-gap recomputation under each flavor ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kball_gap_id", "k_star", "n",
        # HARD (matches the CSV's existing columns)
        "med_HARD", "min_HARD", "mean_HARD",
        # FUZZY_CENTROID
        "med_FC", "min_FC", "mean_FC",
        # FUZZY_JOINTNORM
        "med_FJ", "min_FJ", "mean_FJ",
    ]
    out_rows = []

    for g_col, r in enumerate(targets):
        gap_id = int(r["kball_gap_id"])
        k_star = int(r["refined_k_star"])
        members = np.asarray(
            [int(x) for x in json.loads(r["article_indices_json"])], dtype=np.int64,
        )
        n_members = len(members)
        if n_members < 2:
            continue

        member_emb = np.ascontiguousarray(emb[members])
        D_NN, I_NN = full_index.search(member_emb, k=K_QUERY + 1)
        D_NN = D_NN[:, 1:]
        I_NN = I_NN[:, 1:]
        # Kernel weights from each member to its top-K neighbors
        sigma_seed = sigma[members][:, None]
        sigma_n = sigma[I_NN]
        W = np.exp(-D_NN / (sigma_seed * sigma_n))
        # Mask of "neighbor is a member of THIS gap" (boolean / hard)
        in_G_hard = np.zeros(n_total, dtype=bool)
        in_G_hard[members] = True
        self_mask = (I_NN == members[:, None])
        G_mask_hard = in_G_hard[I_NN] & ~self_mask

        # Local-CSLS-balanced background (same as current pipeline)
        _, I_book = book_index.search(member_emb, k=1)
        nearest_book = book_global_idx[I_book[:, 0]]
        unique_books = np.unique(nearest_book)
        if len(unique_books) > 0:
            book_emb_q = np.ascontiguousarray(emb[unique_books])
            _, I_bart_local = article_index.search(book_emb_q, k=K_QUERY)
            book_to_arts = {
                int(b): article_global_idx[I_bart_local[i]]
                for i, b in enumerate(unique_books)
            }
        else:
            book_to_arts = {}
        from collections import Counter
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
        B_mask = in_B[I_NN]
        K_B = (W * B_mask).sum(axis=1)

        # HARD K_G
        K_G_hard = (W * G_mask_hard).sum(axis=1)
        # FUZZY-CENTROID K_G: weight each neighbor by its centroid-kernel
        # membership in THIS gap (column g_col of M_dense_centroid).
        mu_FC = M_dense_centroid[:, g_col]
        K_G_FC = (W * mu_FC[I_NN] * (~self_mask)).sum(axis=1)
        # FUZZY-JOINTNORM K_G: same, but post-normalization across gaps.
        mu_FJ = M_dense_jointnorm[:, g_col]
        K_G_FJ = (W * mu_FJ[I_NN] * (~self_mask)).sum(axis=1)

        def _stats(K_G_):
            denom = K_G_ + K_B + 1e-30
            post = K_G_ / denom
            # Raw (un-clipped) logits; suppress log-of-zero warnings.
            with np.errstate(divide="ignore", invalid="ignore"):
                logits = np.log(post) - np.log(np.maximum(1.0 - post, 1e-30))
            finite = logits[np.isfinite(logits)]
            mean_v = float(finite.mean()) if finite.size else 0.0
            median_v = float(np.median(logits))
            min_v = float(np.min(logits[np.isfinite(logits)])) if finite.size else 0.0
            return median_v, min_v, mean_v

        med_HARD, min_HARD, mean_HARD = _stats(K_G_hard)
        med_FC, min_FC, mean_FC = _stats(K_G_FC)
        med_FJ, min_FJ, mean_FJ = _stats(K_G_FJ)

        out_rows.append({
            "kball_gap_id": gap_id, "k_star": k_star, "n": n_members,
            "med_HARD": f"{med_HARD:.3f}",
            "min_HARD": f"{min_HARD:.3f}",
            "mean_HARD": f"{mean_HARD:.3f}",
            "med_FC": f"{med_FC:.3f}",
            "min_FC": f"{min_FC:.3f}",
            "mean_FC": f"{mean_FC:.3f}",
            "med_FJ": f"{med_FJ:.3f}",
            "min_FJ": f"{min_FJ:.3f}",
            "mean_FJ": f"{mean_FJ:.3f}",
        })

        print(f"  kb-{gap_id:<4} k*={k_star:<4} n={n_members:<4}"
              f"  HARD med={med_HARD:.2f}/min={min_HARD:.2f}"
              f"  FC med={med_FC:.2f}/min={min_FC:.2f}"
              f"  FJ med={med_FJ:.2f}/min={min_FJ:.2f}",
              flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    print(f"[done] csv written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
