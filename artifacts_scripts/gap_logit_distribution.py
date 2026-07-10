"""Dump per-member logit distribution for selected refined gaps.

For each requested gap (by kball_gap_id), re-runs the local-CSLS posterior
on the refined member set and reports the 11-quantile distribution
(min, q10, q20, ..., q40, median, q60, ..., q90, max) of per-member
logits, plus mean/std in both clipped and unclipped forms.

Quantiles use UN-clipped logits (median etc. are robust to extremes by
sort order). Mean/std are reported BOTH clipped (matching the
mean_logit field in the refined CSV) AND unclipped (raw logit
distribution).

Usage::

    uv run python -u -m artifacts_scripts.gap_logit_distribution \\
        --scope 17/1702/T11303 \\
        --refined-csv artifacts/gap_kball_llr_refine/T11303_n1000__llr_refined.csv \\
        --gap-ids 327,311,588,341,18 \\
        --out cache/gap_logit_distribution/T11303.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from artifacts_scripts.gap_kball_llr_refine import compute_local_csls_posterior
from artifacts_scripts.gap_static_seed_diagnostic import K_QUERY, setup


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True)
    p.add_argument("--refined-csv", required=True, type=Path)
    p.add_argument("--gap-ids", required=True,
                   help="Comma-separated kball_gap_id values to inspect")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    requested_ids = {int(s) for s in args.gap_ids.split(",")}

    refined_rows = []
    with open(args.refined_csv) as f:
        for r in csv.DictReader(f):
            if int(r["kball_gap_id"]) in requested_ids:
                refined_rows.append(r)
    print(f"  found {len(refined_rows)} of {len(requested_ids)} requested gaps",
          flush=True)

    print(f"[setup] loading scope {args.scope}...", flush=True)
    state = setup(args.scope)
    print(f"  N_total={state['sigma'].shape[0]}, "
          f"N_articles={len(state['article_global_idx'])}", flush=True)

    from bookgap.index import build_index, FlatL2
    book_emb = np.ascontiguousarray(state["emb"][state["book_global_idx"]])
    book_index = build_index(FlatL2(), book_emb)

    full_index = state["full_index"]
    sigma = state["sigma"]
    article_index = state["article_index"]
    article_global_idx = state["article_global_idx"]
    book_global_idx = state["book_global_idx"]
    n_total = sigma.shape[0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    quantile_pcts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    fieldnames = [
        "kball_gap_id", "refined_k_star", "n_members",
        "mean_clipped", "std_clipped",
        "mean_raw", "std_raw",
    ] + [f"q{p}" for p in quantile_pcts]

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in refined_rows:
            gap_id = int(r["kball_gap_id"])
            k_star = int(r["refined_k_star"])
            members = json.loads(r["article_indices_json"])
            members_arr = np.asarray(members, dtype=np.int64)
            if len(members_arr) < 2:
                continue

            member_emb = np.ascontiguousarray(state["emb"][members_arr])
            D_NN, I_NN_full = full_index.search(member_emb, k=K_QUERY + 1)
            D_NN = D_NN[:, 1:]
            I_NN_full = I_NN_full[:, 1:]

            _, I_book = book_index.search(member_emb, k=1)
            nearest_book = book_global_idx[I_book[:, 0]]
            unique_books = np.unique(nearest_book)
            if len(unique_books) > 0:
                book_emb_q = np.ascontiguousarray(state["emb"][unique_books])
                _, I_book_arts_local = article_index.search(book_emb_q, k=K_QUERY)
                book_to_arts = {
                    int(b): article_global_idx[I_book_arts_local[r2]]
                    for r2, b in enumerate(unique_books)
                }
            else:
                book_to_arts = {}

            posts = compute_local_csls_posterior(
                members=members,
                D_NN=D_NN.astype(np.float32),
                I_NN=I_NN_full.astype(np.int64),
                sigma=sigma,
                nearest_book_per_member=nearest_book.astype(np.int64),
                book_to_arts=book_to_arts,
                n_total=n_total,
            )

            # Clipped (matches mean_logit in the refined CSV)
            post_clip = np.clip(posts, 1e-3, 1 - 1e-3)
            logits_clip = np.log(post_clip / (1 - post_clip))
            mean_clip = float(logits_clip.mean())
            std_clip = float(logits_clip.std(ddof=0))

            # Raw (no clipping; may contain -inf / +inf for extreme posts)
            with np.errstate(divide="ignore", invalid="ignore"):
                logits_raw = np.log(posts) - np.log(np.maximum(1.0 - posts, 1e-30))
            # For mean/std, replace -inf with very negative finite value to allow finite stats;
            # for quantiles, np.percentile handles -inf by sort order.
            finite_mask = np.isfinite(logits_raw)
            mean_raw = float(logits_raw[finite_mask].mean()) if finite_mask.any() else float("nan")
            std_raw = float(logits_raw[finite_mask].std(ddof=0)) if finite_mask.any() else float("nan")
            # Quantiles on raw (with possible -inf; numpy handles this)
            quantiles = np.quantile(logits_raw, q=[p/100 for p in quantile_pcts])

            row = {
                "kball_gap_id": gap_id,
                "refined_k_star": k_star,
                "n_members": len(members),
                "mean_clipped": f"{mean_clip:.4f}",
                "std_clipped": f"{std_clip:.4f}",
                "mean_raw": f"{mean_raw:.4f}",
                "std_raw": f"{std_raw:.4f}",
            }
            for p_pct, q in zip(quantile_pcts, quantiles):
                row[f"q{p_pct}"] = f"{q:.4f}" if np.isfinite(q) else str(q)
            w.writerow(row)

            print(f"  gap kb-{gap_id} (k*={k_star}, n={len(members)}): "
                  f"mean_clip={mean_clip:.2f} std_clip={std_clip:.2f}  "
                  f"mean_raw={mean_raw:.2f} std_raw={std_raw:.2f}  "
                  f"min={quantiles[0]:.2f} q10={quantiles[1]:.2f} "
                  f"med={quantiles[5]:.2f} q90={quantiles[9]:.2f} max={quantiles[10]:.2f}",
                  flush=True)

    print(f"[done] csv written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
