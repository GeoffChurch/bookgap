"""Multi-cluster softmax Phase 1 comparison.

Reads original soft-EM enriched CSV + new (variant, P, iter, rule)
basin CSVs from gap_softmax_reassign.py, emits Jaccard / eff_rank /
coverage-label / leaderboard diff and a markdown report.

Spec: docs/superpowers/specs/2026-05-08-multicluster-softmax-phase1-design.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_basin_supports(csv_path: Path) -> list[frozenset[int]]:
    """Load a basin CSV -> list of frozensets of article global indices.

    Uses the member_indices_json column. Empty basins (n_articles=0)
    yield empty frozensets in the same position.
    """
    out: list[frozenset[int]] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            members = json.loads(row["member_indices_json"])
            out.append(frozenset(int(m) for m in members))
    return out


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def best_match_jaccard(
    src: list[frozenset[int]],
    tgt: list[frozenset[int]],
) -> dict:
    """For each non-empty basin in src, find the best Jaccard against
    tgt. Return distribution stats.

    Empty basins are skipped (they have no support; not informative).
    """
    scores: list[float] = []
    for s in src:
        if not s:
            continue
        scores.append(max((_jaccard(s, t) for t in tgt), default=0.0))
    arr = np.array(scores, dtype=np.float64)
    if len(arr) == 0:
        return {"count": 0, "mean": float("nan"), "p25": float("nan"),
                "p50": float("nan"), "p75": float("nan"), "p95": float("nan"),
                "count_ge_0.95": 0, "count_lt_0.80": 0, "count_eq_1.0": 0}
    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "count_ge_0.95": int((arr >= 0.95).sum()),
        "count_lt_0.80": int((arr < 0.80).sum()),
        "count_eq_1.0": int((arr == 1.0).sum()),
    }


def effective_rank(p: np.ndarray) -> np.ndarray:
    """exp(H(p_row)) per row. p assumed nonneg, rows sum to 1.

    Returns shape (N,) float64. exp(0) = 1 for pure rows.
    """
    p_clip = np.clip(p, 1e-30, 1.0)
    H = -(p_clip * np.log(p_clip)).sum(axis=1)
    return np.exp(H)


def argmax_shift_rate(M_old: np.ndarray, M_new: np.ndarray) -> float:
    """Fraction of rows whose argmax column index differs."""
    return float((M_old.argmax(axis=1) != M_new.argmax(axis=1)).mean())


def eff_rank_summary(eff_old: np.ndarray, eff_new: np.ndarray) -> dict:
    delta = eff_old - eff_new   # positive = softmax sharpens
    return {
        "median_old": float(np.median(eff_old)),
        "median_new": float(np.median(eff_new)),
        "median_delta": float(np.median(delta)),
        "p25_delta": float(np.percentile(delta, 25)),
        "p75_delta": float(np.percentile(delta, 75)),
        "iqr_delta": float(np.percentile(delta, 75) - np.percentile(delta, 25)),
        "frac_delta_ge_0.5": float((delta >= 0.5).mean()),
        "frac_delta_le_neg_0.5": float((delta <= -0.5).mean()),
    }


def save_eff_rank_scatter(
    out_path: Path,
    eff_old: np.ndarray,
    eff_new_oneshot: np.ndarray | None,
    eff_new_iterated: np.ndarray | None,
    title_suffix: str = "",
) -> None:
    """2D hexbin: x = eff_rank_old, y = eff_rank_new. Two panels:
    one-shot and iterated, side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("one-shot", eff_new_oneshot),
        ("iterated", eff_new_iterated),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, (name, eff_new) in zip(axes, panels):
        if eff_new is None:
            ax.text(0.5, 0.5, f"({name} missing)", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        ax.hexbin(eff_old, eff_new, gridsize=30, mincnt=1, cmap="viridis")
        max_val = max(eff_old.max(), eff_new.max())
        ax.plot([1, max_val], [1, max_val], "r--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{name}{title_suffix}")
        ax.set_xlabel("eff_rank old (normalized soft-EM)")
        ax.set_ylabel("eff_rank new (multi-cluster softmax)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def coverage_label_v2_minimal(row: dict, inh: dict | None) -> str:
    """Minimal port of coverage_label_v2 from gap_multi_scale_coverage.py.

    Tier order (descending priority):
        templated-leak > book-anchored > book-exists > inherited-strong
        > inherited-moderate > inherited-weak > real-gap

    Inputs are CSV row dicts with string values (matching enriched CSV).
    `inh` is a per-basin dict from coverage post-processing or None.
    """
    if int(row.get("templated_leak_flag", 0) or 0):
        return "templated-leak"
    if int(row.get("n_books", 0) or 0) > 0:
        return "book-anchored"
    if int(row.get("book_exists_flag", 0) or 0):
        return "book-exists"
    if inh and inh.get("max_basin_rank", ""):
        mp_str = inh.get("median_post", "")
        if mp_str:
            try:
                mp = float(mp_str)
            except ValueError:
                mp = 0.0
            if mp >= 0.75:
                return "inherited-strong"
            if mp >= 0.65:
                return "inherited-moderate"
            if mp >= 0.55:
                return "inherited-weak"
    return "real-gap"


def coverage_label_distribution(
    rows: list[dict],
    inh_by_idx: dict[int, dict],
) -> dict[str, int]:
    """Count coverage_label_v2_minimal outputs across rows."""
    counts: dict[str, int] = {}
    for i, row in enumerate(rows):
        label = coverage_label_v2_minimal(row, inh_by_idx.get(i))
        counts[label] = counts.get(label, 0) + 1
    return counts


def top_n_symdiff(
    old_top: list[frozenset[int]],
    new_top: list[frozenset[int]],
    n: int = 20,
) -> dict[str, list[frozenset[int]]]:
    """Symmetric difference of top-n basin frozensets."""
    old_set = set(old_top[:n])
    new_set = set(new_top[:n])
    return {
        "only_old": list(old_set - new_set),
        "only_new": list(new_set - old_set),
        "in_both": list(old_set & new_set),
    }


def compute_effective_uncovered_new(
    M: np.ndarray,
    basin_meta: list[dict],
) -> np.ndarray:
    """Direct softmax readout per the parent design doc:

        effective_uncovered(G) = Σ_a p(a∈G) · (1 − Σ_H∈book-anchored p(a∈H))

    Args:
        M: (N_articles, N_basins) softmax membership matrix.
        basin_meta: list of dicts with `book_anchored` flag.

    Returns:
        (N_basins,) float64.
    """
    book_anchored_mask = np.array(
        [bool(m.get("book_anchored", 0)) for m in basin_meta], dtype=bool
    )
    if book_anchored_mask.any():
        coverage_per_article = M[:, book_anchored_mask].sum(axis=1)
    else:
        coverage_per_article = np.zeros(M.shape[0], dtype=np.float64)
    coverage_per_article = np.clip(coverage_per_article, 0.0, 1.0)
    uncovered_per_article = 1.0 - coverage_per_article
    return (M * uncovered_per_article[:, None]).sum(axis=0)


def _read_old_supports_and_meta(
    enriched_csv: Path,
) -> tuple[list[frozenset[int]], list[dict]]:
    """Load original soft-EM enriched CSV -> supports + per-basin meta dicts."""
    supports: list[frozenset[int]] = []
    meta: list[dict] = []
    with enriched_csv.open() as f:
        for row in csv.DictReader(f):
            members = json.loads(row["member_indices_json"])
            supports.append(frozenset(int(m) for m in members))
            meta.append(dict(row))
    return supports, meta


def _normalize_old_M_per_article(
    enriched_csv: Path,
    article_global_idx_to_local: dict[int, int],
    n_articles: int,
    n_basins: int,
) -> np.ndarray:
    """Build a normalized old soft-EM membership matrix:
        p_old(a∈G_k) = w_old(a, k) / Σ_j w_old(a, j)
    Returns (N_articles, N_basins) float32 with rows summing to 1
    where any basin claimed the article, else 0.
    """
    M_raw = np.zeros((n_articles, n_basins), dtype=np.float32)
    with enriched_csv.open() as f:
        for k, row in enumerate(csv.DictReader(f)):
            weights = json.loads(row["member_weights_json"])
            for w in weights:
                if w["type"] != "article":
                    continue
                local = article_global_idx_to_local.get(int(w["idx"]))
                if local is None:
                    continue
                M_raw[local, k] = float(w["weight"])
    row_sum = M_raw.sum(axis=1, keepdims=True)
    M_norm = np.where(row_sum > 0, M_raw / np.where(row_sum > 0, row_sum, 1.0), 0.0)
    return M_norm.astype(np.float32)


def _render_report_markdown(rows: list[dict], scope: str) -> str:
    """Render comparison_report.md from the rows."""
    lines: list[str] = []
    lines.append(f"# Multi-cluster softmax Phase 1 — {scope} comparison report\n")
    lines.append(f"_Generated by gap_softmax_compare.py._\n")
    lines.append("## TL;DR table\n")
    if not rows:
        lines.append("(no rows)\n")
        return "\n".join(lines)

    headers = [
        "variant", "P", "iter_mode", "rule", "iters", "converged",
        "final_delta", "achieved_P_last",
        "jaccard_old_to_new_mean", "jaccard_new_to_old_mean",
        "median_eff_rank_delta",
        "real_gap_count_new", "argmax_shift_rate",
        "jaccard_old_to_new_count_lt_0.80",
        "top20_only_new_count",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        cells = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Notes on metrics\n")
    lines.append("- **jaccard_old_to_new_mean**: mean over old basins of "
                 "max-Jaccard against any new basin under this rule.")
    lines.append("- **jaccard_new_to_old_mean**: reverse direction.")
    lines.append("- **real_gap_count_new** uses a minimal label proxy: "
                 "any non-book-anchored basin counts as real-gap. The richer "
                 "tiering (inherited-strong/moderate/weak via "
                 "gap_multi_scale_coverage) is out of Phase 1 scope.")
    lines.append("- **top20_only_new_count**: number of new top-20 basins "
                 "(by direct softmax effective_uncovered readout) absent "
                 "from the old top-20 (by Phase-3 effective_uncovered).")
    lines.append("- **achieved_P_last**: realized perplexity at the final "
                 "tau-tuning step; should be close to P_target.")
    lines.append("")

    lines.append("## Synthesis\n")
    lines.append("(Fill in by hand: which configs reproduce original basins, "
                 "which diverge, whether iteration drives the divergence, "
                 "whether empty basins matter.)\n")

    lines.append("## Phase 2 verdict\n")
    lines.append("(Fill in by hand: choose one of)\n")
    lines.append("- **Phase 2 not justified** — one-shot reproduces; iterated stays close.")
    lines.append("- **Phase 2 justified by formulation** — one-shot already differs materially.")
    lines.append("- **Phase 2 justified by iteration** — one-shot approx original; iterated diverges.")
    lines.append("")

    lines.append("## Recommendation for T11303\n")
    lines.append("(Fill in by hand: skip if verdict is clear; run if ambiguous.)\n")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True)
    p.add_argument("--enriched-csv", required=True,
                   help="Original soft-EM enriched CSV.")
    p.add_argument("--reassign-dir", required=True,
                   help="Directory containing memberships__*.npz, basins__*.csv, meta__*.json.")
    p.add_argument("--output-dir", required=True,
                   help="Where to write comparison_table.csv, comparison_report.md, scatter PNGs.")
    args = p.parse_args()

    from artifacts_scripts.hdbscan_gap_probe import load_scope
    from scipy.sparse import load_npz

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reassign_dir = Path(args.reassign_dir)

    emb, ids, types, _titles, _topic_ids, _refs = load_scope(args.scope, with_refs=False)
    is_article = np.array(types) == "article"
    article_global_idx = np.where(is_article)[0]
    article_global_idx_to_local = {int(g): l for l, g in enumerate(article_global_idx)}
    n_articles = len(article_global_idx)

    old_supports, old_meta = _read_old_supports_and_meta(Path(args.enriched_csv))
    n_basins = len(old_supports)
    M_old_norm = _normalize_old_M_per_article(
        Path(args.enriched_csv), article_global_idx_to_local, n_articles, n_basins,
    )
    eff_rank_old = effective_rank(np.where(M_old_norm.sum(axis=1, keepdims=True) > 0,
                                            M_old_norm,
                                            np.full_like(M_old_norm, 1.0 / n_basins)))

    rows = []
    eff_new_by_variant_iter: dict[tuple[str, str, float], np.ndarray] = {}

    for memb_path in sorted(reassign_dir.glob("memberships__*.npz")):
        config_tag = memb_path.stem.replace("memberships__", "")
        parts = config_tag.split("__")
        variant = parts[0].replace("variant", "")
        P = float(parts[1].replace("P", ""))
        iter_mode = parts[2]

        meta_path = reassign_dir / f"meta__{config_tag}.json"
        meta = json.loads(meta_path.read_text())

        M_new = load_npz(memb_path).toarray().astype(np.float32)
        if M_new.shape != (n_articles, n_basins):
            raise ValueError(f"shape mismatch: {memb_path} has {M_new.shape}, expected ({n_articles},{n_basins})")
        eff_rank_new = effective_rank(M_new + 1e-30)

        eff_new_by_variant_iter[(variant, iter_mode, P)] = eff_rank_new

        am_shift = argmax_shift_rate(M_old_norm, M_new)
        er_summary = eff_rank_summary(eff_rank_old, eff_rank_new)

        eff_unc_new = compute_effective_uncovered_new(M_new, old_meta)
        eff_unc_old = np.array(
            [float(r.get("effective_uncovered_articles", "0") or 0) for r in old_meta],
            dtype=np.float64,
        )
        n_new = np.argsort(-eff_unc_new)
        n_old = np.argsort(-eff_unc_old)
        argmax_per_article = M_new.argmax(axis=1)
        new_supports_argmax: list[frozenset[int]] = []
        for k in range(n_basins):
            mems = np.where(argmax_per_article == k)[0]
            new_supports_argmax.append(frozenset(int(article_global_idx[l]) for l in mems))
        old_top_supports = [old_supports[k] for k in n_old]
        new_top_supports = [new_supports_argmax[k] for k in n_new]
        ldb = top_n_symdiff(old_top_supports, new_top_supports, n=20)

        for rule in ["argmax", "p_gt_0.5", "confidence_margin"]:
            csv_path = reassign_dir / f"basins__{config_tag}__{rule}.csv"
            if not csv_path.exists():
                continue
            new_supports = load_basin_supports(csv_path)
            jaccard_old_to_new = best_match_jaccard(old_supports, new_supports)
            jaccard_new_to_old = best_match_jaccard(new_supports, old_supports)

            label_dist_new = {"book-anchored": 0, "real-gap": 0}
            with csv_path.open() as f:
                for nrow in csv.DictReader(f):
                    if int(nrow.get("n_articles", 0) or 0) == 0:
                        continue
                    if int(nrow.get("book_anchored", 0) or 0):
                        label_dist_new["book-anchored"] += 1
                    else:
                        label_dist_new["real-gap"] += 1

            label_dist_old = coverage_label_distribution(old_meta, inh_by_idx={})

            rows.append({
                "variant": variant,
                "P": P,
                "iter_mode": iter_mode,
                "rule": rule,
                "iters": meta.get("iters", -1),
                "converged": meta.get("converged", False),
                "final_delta": meta.get("final_delta", float("nan")),
                "achieved_P_last": meta.get("achieved_Ps", [float("nan")])[-1],
                "argmax_shift_rate": am_shift,
                "jaccard_old_to_new_mean": jaccard_old_to_new["mean"],
                "jaccard_old_to_new_count_ge_0.95": jaccard_old_to_new["count_ge_0.95"],
                "jaccard_old_to_new_count_lt_0.80": jaccard_old_to_new["count_lt_0.80"],
                "jaccard_new_to_old_mean": jaccard_new_to_old["mean"],
                "jaccard_new_to_old_count_ge_0.95": jaccard_new_to_old["count_ge_0.95"],
                "jaccard_new_to_old_count_lt_0.80": jaccard_new_to_old["count_lt_0.80"],
                "median_eff_rank_delta": er_summary["median_delta"],
                "iqr_eff_rank_delta": er_summary["iqr_delta"],
                "real_gap_count_old": label_dist_old.get("real-gap", 0),
                "real_gap_count_new": label_dist_new.get("real-gap", 0),
                "top20_only_old_count": len(ldb["only_old"]),
                "top20_only_new_count": len(ldb["only_new"]),
            })

    table_path = out_dir / "comparison_table.csv"
    if rows:
        with table_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {table_path} ({len(rows)} rows)", flush=True)

    for variant in ("A", "B"):
        eff_oneshot = eff_new_by_variant_iter.get((variant, "oneshot", 2.0))
        eff_iterated = eff_new_by_variant_iter.get((variant, "iterated", 2.0))
        if eff_oneshot is None and eff_iterated is None:
            continue
        save_eff_rank_scatter(
            out_dir / f"eff_rank_scatter__variant{variant}.png",
            eff_rank_old, eff_oneshot, eff_iterated,
            title_suffix=f" (variant {variant}, P=2.0)",
        )
        print(f"Wrote eff_rank_scatter__variant{variant}.png", flush=True)

    report_path = out_dir / "comparison_report.md"
    with report_path.open("w") as f:
        f.write(_render_report_markdown(rows, args.scope))
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
