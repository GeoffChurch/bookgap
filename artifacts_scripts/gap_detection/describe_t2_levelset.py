"""Descriptive pass: WHAT were the level-set components that filled (Track 2 wiki)?

Recomputes the phase-3 levelset predictor deterministically from the frozen phase-2
caches (same code path as run_t2_phase3), then prints — titles only, no article text —
each component that entered the top-50 budget: size, n_eff, per-component fill vs the
scoped-pool median, example T-member titles, and example (T, T+Δ] arrival titles that
landed inside the component's neighborhoods.

Usage: uv run python -m artifacts_scripts.gap_detection.describe_t2_levelset
"""

from __future__ import annotations

import numpy as np

from artifacts_scripts.gap_detection.run_t2_phase3 import (
    K_NN,
    TOP_K,
    GapScorer,
    fill_counts,
    knn_scale,
    level_set_components,
    load_phase2,
)


def main() -> None:
    d, K_corpus, arr, grow, models, meta = load_phase2()
    Z = d["Z"]
    fg_mask = d["fg_mask"]
    scoped_idx = d["scoped_idx"]
    titles = d["titles"]
    Zf = Z[fg_mask]
    Zb_s = Z[~fg_mask][scoped_idx]
    bg_rows_global = np.where(~fg_mask)[0]
    scoped_global = bg_rows_global[scoped_idx]
    Z_arr = arr["Z"]
    arr_titles = arr["titles"]

    scorer = GapScorer(bg=Zb_s, probes=None, k=K_NN)
    scorer.set_r0_from_fg(Zf)
    _ = scorer.n_Bloc
    out = scorer.score(Zf)

    comps = level_set_components(out["full"], scorer.cand, scorer.real_mask, top_frac=0.05)

    # Pool context: median per-candidate fill over the scoped pool (outcome side).
    pool_radii = knn_scale(Zb_s, Zb_s, k=K_NN, self_rows=np.ones(Zb_s.shape[0], dtype=bool))
    pool_fill = fill_counts(Zb_s, pool_radii, Z_arr)
    print(f"scoped pool: {Zb_s.shape[0]:,} candidates, median fill "
          f"{np.median(pool_fill):.0f}, mean {pool_fill.mean():.1f} arrivals/neighborhood\n")

    budget = TOP_K
    rank = 0
    for c in comps[:8]:  # show top components even past the predictor's 50-doc budget
        rank += 1
        members = np.asarray(c["members"])
        order = members[np.argsort(out["full"][members])[::-1]]
        take = order[:max(budget, 8)]
        budget -= len(take)

        radii = pool_radii[take]
        fills = pool_fill[take]
        # arrival examples: nearest arrivals within radius of the top member rows
        ex_arrivals: list[str] = []
        seen = set()
        X = Zb_s[take]
        # chunked distance to keep memory flat
        for i in range(len(take)):
            dist = np.linalg.norm(Z_arr - X[i], axis=1)
            inside = np.argsort(dist)[:8]
            for j in inside:
                if dist[j] <= radii[i] and j not in seen:
                    seen.add(j)
                    ex_arrivals.append(str(arr_titles[j]))
            if len(ex_arrivals) >= 8:
                break

        member_titles = [str(titles[scoped_global[m]]) for m in order[:8]]
        print(f"=== component #{rank}  size={len(members)}  n_eff={c.get('n_eff', float('nan')):.1f}  "
              f"in-budget={len(take)}  fill/member median={np.median(fills):.0f} "
              f"(pool median {np.median(pool_fill):.0f}) ===")
        print(f"  T-members  : {'; '.join(member_titles)}")
        print(f"  arrivals → : {'; '.join(ex_arrivals[:8]) if ex_arrivals else '(none inside radii)'}")
        print()


if __name__ == "__main__":
    main()
