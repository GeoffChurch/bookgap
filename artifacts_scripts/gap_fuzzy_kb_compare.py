"""Compare hard K_B vs fuzzy-complement K_B for refined gaps.

Builds on the FJ K_G default from gap_kball_llr_refine. The question:
once K_G is fuzzy (joint-normalized), does the K_B side need to be
fuzzy too for the posterior to feel "symmetric"?

The fuzzy-complement K_B formulation:

  K_B_fuzzy(i) = Σ_{j ∈ B_CSLS} W_ij · (1 - μ_G_thisgap(j))

where μ_G_thisgap(j) is article j's fuzzy membership in THIS gap.
For this to actually differ from hard K_B, μ_G must be nonzero on
B_CSLS articles — which only happens if the joint-normalized
membership matrix covers articles BEYOND each gap's top-k* hard
member set.

So this script extends the M_centroid matrix to ALL top-K_QUERY
articles per gap (not just members), joint-normalizes, and then
recomputes per-member posteriors and median/min logit under four
combinations:

  - HARD K_G + HARD K_B (the pre-Phase-3.5 default)
  - FJ K_G + HARD K_B (Phase 3.5 current default)
  - HARD K_G + FUZZY K_B (asymmetric the other way)
  - FJ K_G + FUZZY K_B (fully symmetric)

Usage::

    .conda/bin/python -u -m artifacts_scripts.gap_fuzzy_kb_compare \\
        --scope 17/1702/T11303 \\
        --refined-csv artifacts/gap_kball_llr_refine/T11303_n1000__llr_refined.csv \\
        --out cache/gap_fuzzy_kb_compare/T11303_top25.csv \\
        --top-n 25
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, diags

from artifacts_scripts.gap_static_seed_diagnostic import K_QUERY, setup


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True)
    p.add_argument("--refined-csv", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--extra-gap-ids", default="",
                   help="Comma-separated kball_gap_ids to ALSO include, "
                        "even if not in top-N (e.g., for inspecting pruned gaps)")
    args = p.parse_args()

    print(f"[load] reading {args.refined_csv}...", flush=True)
    all_rows = []
    with open(args.refined_csv) as f:
        for r in csv.DictReader(f):
            all_rows.append(r)
    survivors = [r for r in all_rows if r["survivor"] == "1"]
    survivors.sort(key=lambda r: -float(r["median_logit"]))
    targets = survivors[: args.top_n]
    extra_ids = {int(x) for x in args.extra_gap_ids.split(",") if x.strip()}
    target_ids = {int(t["kball_gap_id"]) for t in targets}
    for r in all_rows:
        gid = int(r["kball_gap_id"])
        if gid in extra_ids and gid not in target_ids:
            targets.append(r)
            target_ids.add(gid)
    print(f"  {len(all_rows)} rows, {len(survivors)} survivors, "
          f"analyzing {len(targets)} gaps (top-{args.top_n} + "
          f"{len(extra_ids)} extras)", flush=True)

    print(f"[setup] loading scope {args.scope}...", flush=True)
    state = setup(args.scope)
    n_total = state["sigma"].shape[0]
    print(f"  N_total={n_total}", flush=True)

    from bookgap.index import build_index, FlatL2
    book_emb = np.ascontiguousarray(state["emb"][state["book_global_idx"]])
    book_index = build_index(FlatL2(), book_emb)
    article_index = state["article_index"]
    article_global_idx = state["article_global_idx"]
    book_global_idx = state["book_global_idx"]
    emb = state["emb"]
    sigma = state["sigma"]
    full_index = state["full_index"]

    # --- Build EXTENDED M_centroid covering top-K_QUERY articles per gap ---
    # This is what enables fuzzy K_B: each gap claims partial membership
    # over its k_query-NN to centroid, not just its top-k* hard members.
    print(f"[fuzzy] building extended M (top-{K_QUERY} per gap centroid)...",
          flush=True)
    rows_idx, cols_idx, vals = [], [], []
    # Track per-gap top-K_QUERY for K_G computation: (gap_idx → (I_topK_global, w_topK))
    gap_topk_neighbors: list[tuple[np.ndarray, np.ndarray]] = []
    for g_idx, r in enumerate(targets):
        members = np.asarray(
            [int(x) for x in json.loads(r["article_indices_json"])], dtype=np.int64,
        )
        member_emb = emb[members]
        centroid = member_emb.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-12:
            # Degenerate; fall back to hard membership over members only.
            gap_topk_neighbors.append((members, np.ones(len(members))))
            for m in members:
                rows_idx.append(int(m))
                cols_idx.append(g_idx)
                vals.append(1.0)
            continue
        centroid_unit = (centroid / norm).astype(np.float32)
        # Query top-K_QUERY nearest articles to centroid.
        D_centroid, I_centroid_local = article_index.search(
            centroid_unit[None, :], k=K_QUERY,
        )
        D_centroid = D_centroid[0]
        I_centroid_global = article_global_idx[I_centroid_local[0]]
        # Centroid sigma: mean over gap members (proxy since centroid is virtual)
        sigma_centroid = float(sigma[members].mean())
        sigma_neighbors = sigma[I_centroid_global]
        w_centroid = np.exp(-D_centroid / (sigma_centroid * sigma_neighbors))
        gap_topk_neighbors.append((I_centroid_global, w_centroid))
        for n_g, w_n in zip(I_centroid_global, w_centroid):
            rows_idx.append(int(n_g))
            cols_idx.append(g_idx)
            vals.append(float(w_n))

    M = csr_matrix((vals, (rows_idx, cols_idx)), shape=(n_total, len(targets)))
    row_sums = np.asarray(M.sum(axis=1)).flatten()
    inv = np.where(row_sums > 0, 1.0 / np.where(row_sums > 0, row_sums, 1.0), 0.0)
    M_norm = (diags(inv) @ M).tocsc()
    print(f"  M shape={M.shape}, nnz={M.nnz}", flush=True)

    # --- For each target gap, compute K_G and K_B under all 4 combinations ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kball_gap_id", "k_star", "n",
        "med_HH", "min_HH",  # HARD K_G + HARD K_B
        "med_FH", "min_FH",  # FJ K_G + HARD K_B
        "med_HF", "min_HF",  # HARD K_G + FUZZY K_B
        "med_FF", "min_FF",  # FJ K_G + FUZZY K_B
    ]
    out_rows = []

    for g_idx, r in enumerate(targets):
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

        # Per-member kernel weights
        sigma_seed = sigma[members][:, None]
        sigma_n = sigma[I_NN]
        W = np.exp(-D_NN / (sigma_seed * sigma_n))
        self_mask = I_NN == members[:, None]

        # Hard gap membership
        in_G_hard = np.zeros(n_total, dtype=bool)
        in_G_hard[members] = True

        # Fuzzy joint-normalized membership for THIS gap
        mu_FJ = np.asarray(M_norm[:, g_idx].todense()).flatten()

        # CSLS-balanced hard background (same as production)
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

        # K_G variants
        K_G_hard = (W * (in_G_hard[I_NN] & ~self_mask).astype(np.float64)).sum(axis=1)
        K_G_FJ = (W * mu_FJ[I_NN] * (~self_mask)).sum(axis=1)

        # K_B variants
        B_mask = in_B[I_NN]
        K_B_hard = (W * B_mask).sum(axis=1)
        # Fuzzy K_B: weight each background article's contribution by (1 - μ_G_thisgap)
        K_B_fuzzy = (W * B_mask * (1.0 - mu_FJ[I_NN])).sum(axis=1)

        def _stats(K_G_, K_B_):
            denom = K_G_ + K_B_ + 1e-30
            post = K_G_ / denom
            with np.errstate(divide="ignore", invalid="ignore"):
                logits = np.log(post) - np.log(np.maximum(1.0 - post, 1e-30))
            median_v = float(np.median(logits))
            finite = logits[np.isfinite(logits)]
            min_v = float(finite.min()) if finite.size else 0.0
            return median_v, min_v

        med_HH, min_HH = _stats(K_G_hard, K_B_hard)
        med_FH, min_FH = _stats(K_G_FJ, K_B_hard)
        med_HF, min_HF = _stats(K_G_hard, K_B_fuzzy)
        med_FF, min_FF = _stats(K_G_FJ, K_B_fuzzy)

        out_rows.append({
            "kball_gap_id": gap_id, "k_star": k_star, "n": n_members,
            "med_HH": f"{med_HH:.3f}", "min_HH": f"{min_HH:.3f}",
            "med_FH": f"{med_FH:.3f}", "min_FH": f"{min_FH:.3f}",
            "med_HF": f"{med_HF:.3f}", "min_HF": f"{min_HF:.3f}",
            "med_FF": f"{med_FF:.3f}", "min_FF": f"{min_FF:.3f}",
        })
        print(f"  kb-{gap_id:<4} k*={k_star:<4} n={n_members:<4}"
              f"  HH med={med_HH:.2f}/min={min_HH:.2f}"
              f"  FH med={med_FH:.2f}/min={min_FH:.2f}"
              f"  HF med={med_HF:.2f}/min={min_HF:.2f}"
              f"  FF med={med_FF:.2f}/min={min_FF:.2f}",
              flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    print(f"[done] csv written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
