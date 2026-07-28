"""Funding-WASTE spike: analysis. Reads cache/dataset.json (built by collect.py).

Probe:
  dollars  = total_award                       (OPERATIONAL, want HIGH)
  output   = n_pub / sum_citations / sum_rcr   (STRUCTURAL footprint, want LOW)
  waste    = high dollars + low output  (inverse of gap score Delta)

Validity test = the whole point:
  - RAW relationship: regress log(output+1) ~ log(dollars+1); rank by NEG residual.
  - AGE-CONTROLLED: add grant-age covariates (first_fy, years_since_start) and
    re-examine the partial relationship of dollars->output. Is there RESIDUAL
    output-per-dollar variation, or is "funded-but-invisible" explained by age?
  - MECHANISM: we restrict to R01 (a publishing mechanism) up front. We also
    report how flagged grants split on ever_supplement / single-year vs renewal.

Regenerate:
    uv run \
        python -m artifacts_scripts.funding_spike.analyze
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
DATA_REF_YEAR = 2026  # citations observed as of ~now


def _start_year(s: str | None) -> int | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "")).year
    except Exception:
        return None


def partial_corr(x, y, z):
    """Partial correlation of x,y controlling for columns in z (n x k)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    Z = np.column_stack([np.ones(len(x))] + [np.asarray(c, float) for c in z])
    # residualize x and y on Z
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    rx = x - Z @ bx
    ry = y - Z @ by
    r, p = stats.pearsonr(rx, ry)
    return r, p, rx, ry


def ols(y, X_cols, names):
    """Plain OLS with t-stats. X_cols: list of 1d arrays (no intercept col)."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X_cols])
    names = ["intercept"] + names
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = n - k
    sigma2 = (resid @ resid) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t = beta / se
    pvals = 2 * stats.t.sf(np.abs(t), dof)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot
    return {
        "names": names, "beta": beta, "se": se, "t": t, "p": pvals,
        "r2": r2, "resid": resid, "n": n, "dof": dof,
    }


def main():
    data = json.loads((CACHE / "dataset.json").read_text())
    rows = data
    n_all = len(rows)

    # derive grant age (years since project start, as of ref year)
    for r in rows:
        sy = _start_year(r.get("start_date")) or r.get("first_fy")
        r["start_year"] = sy
        r["age"] = (DATA_REF_YEAR - sy) if sy else None

    # keep grants with a usable start year and positive dollars
    rows = [r for r in rows if r["age"] is not None and r["total_award"] and r["total_award"] > 0]
    n_used = len(rows)

    dollars = np.array([r["total_award"] for r in rows], float)
    n_pub = np.array([r["n_pub"] for r in rows], float)
    cites = np.array([r["sum_citations"] for r in rows], float)
    rcr = np.array([r["sum_rcr"] for r in rows], float)
    age = np.array([r["age"] for r in rows], float)
    first_fy = np.array([r["first_fy"] for r in rows], float)
    n_fy = np.array([r["n_fy_rows"] for r in rows], float)

    ld = np.log(dollars + 1)
    # three output proxies
    outputs = {
        "n_pub": n_pub,
        "sum_citations": cites,
        "sum_rcr": rcr,
    }

    report = {}
    report["n_all_grants"] = n_all
    report["n_used_grants"] = n_used
    report["filters"] = {
        "mechanism": "R01 only",
        "fiscal_years_first_award": "2013-2016",
        "selected": "random dollar-neutral subsample (seed in collect.py)",
        "ref_year_for_citations": DATA_REF_YEAR,
    }

    # Robustness: restrict to OLD grants (age>=15) so the time-lag confound is
    # largely removed. If a waste signal exists it should show as residual
    # output-per-dollar variation here too (and the partial dollar->output corr
    # should NOT flip negative).
    old_mask = age >= 15
    if old_mask.sum() > 50:
        ld_o = ld[old_mask]
        for nm, yv in [("n_pub", n_pub), ("sum_citations", cites), ("sum_rcr", rcr)]:
            ly_o = np.log(yv[old_mask] + 1)
            r, p, _, _ = partial_corr(ld_o, ly_o,
                                      [age[old_mask], first_fy[old_mask], n_fy[old_mask]])
            report.setdefault("controlled_old_only", {})[nm] = {
                "n_old": int(old_mask.sum()),
                "partial_corr_dollar_output": [float(r), float(p)],
                "frac_zero_pub_old": float((n_pub[old_mask] == 0).mean()),
            }
    report["dollars_summary"] = {
        "min": float(dollars.min()), "median": float(np.median(dollars)),
        "mean": float(dollars.mean()), "max": float(dollars.max()),
    }
    report["zero_output"] = {
        "n_zero_pub": int((n_pub == 0).sum()),
        "frac_zero_pub": float((n_pub == 0).mean()),
        "n_zero_cite": int((cites == 0).sum()),
        "frac_zero_cite": float((cites == 0).mean()),
    }

    # ---- RAW relationship: log(output+1) ~ log(dollars+1) ----
    report["raw"] = {}
    for name, y in outputs.items():
        ly = np.log(y + 1)
        r, p = stats.pearsonr(ld, ly)
        sp, spp = stats.spearmanr(dollars, y)
        report["raw"][name] = {
            "pearson_logdollar_logoutput": [float(r), float(p)],
            "spearman_dollar_output": [float(sp), float(spp)],
        }

    # ---- AGE/MECHANISM-controlled (partial) relationship ----
    # control vars: age, first_fy (proxy of cohort), n_fy_rows (renewal length /
    # how long the grant ran -> more years = more time + money to publish).
    report["controlled"] = {}
    controls = [age, first_fy, n_fy]
    control_names = ["age", "first_fy", "n_fy_rows"]
    for name, y in outputs.items():
        ly = np.log(y + 1)
        # partial corr of dollars vs output controlling for age cohort + duration
        r, p, rx, ry = partial_corr(ld, ly, controls)
        report["controlled"][name] = {
            "partial_corr_dollar_output_given_age_fy_dur": [float(r), float(p)],
        }

    # ---- full OLS for log(n_pub+1): does dollars retain a coefficient after age? ----
    for name, y in outputs.items():
        ly = np.log(y + 1)
        fit = ols(ly, [ld, age, first_fy, n_fy], ["log_dollars", "age", "first_fy", "n_fy_rows"])
        report.setdefault("ols", {})[name] = {
            "names": fit["names"],
            "beta": [float(b) for b in fit["beta"]],
            "t": [float(t) for t in fit["t"]],
            "p": [float(pp) for pp in fit["p"]],
            "r2": float(fit["r2"]), "n": fit["n"],
        }

    # ---- residual ranking: who is funded-but-invisible AFTER controlling? ----
    # Use log(sum_citations+1) as the structural footprint oracle (richer than
    # raw pub count). Residual from regressing it on age+first_fy+n_fy+log_dollars
    # would partial OUT dollars; instead we want: given dollars & age, how much
    # LESS output than expected? So regress output on age+first_fy+n_fy+log_dollars
    # and take the most-negative residual = produced far less than peers of equal
    # money AND equal age.
    y = np.log(cites + 1)
    fit = ols(y, [ld, age, first_fy, n_fy], ["log_dollars", "age", "first_fy", "n_fy_rows"])
    resid = fit["resid"]
    order = np.argsort(resid)  # most negative first
    # also: simplest possible flag — biggest dollars among zero-pub grants
    zero_idx = [i for i in range(len(rows)) if n_pub[i] == 0]
    zero_idx.sort(key=lambda i: -dollars[i])

    def grant_brief(i):
        r = rows[i]
        return {
            "core_project_num": r["core_project_num"],
            "total_award": int(r["total_award"]),
            "first_fy": r["first_fy"],
            "fiscal_years": r["fiscal_years"],
            "start_year": r["start_year"],
            "age": int(r["age"]),
            "n_fy_rows": r["n_fy_rows"],
            "n_pub": int(r["n_pub"]),
            "sum_citations": int(r["sum_citations"]),
            "sum_rcr": round(r["sum_rcr"], 2),
            "appl_types": r["appl_types"],
            "ever_supplement": r["ever_supplement"],
            "resid_logcite": round(float(resid[i]), 3),
            "title": (r["title"] or "")[:120],
        }

    report["top_residual_flags"] = [grant_brief(i) for i in order[:12]]
    report["top_zero_pub_by_dollars"] = [grant_brief(i) for i in zero_idx[:12]]

    # characterize zero-pub grants vs the rest on age & supplement
    zero_mask = n_pub == 0
    if zero_mask.any() and (~zero_mask).any():
        report["zero_vs_nonzero"] = {
            "mean_age_zero": float(age[zero_mask].mean()),
            "mean_age_nonzero": float(age[~zero_mask].mean()),
            "mean_dollars_zero": float(dollars[zero_mask].mean()),
            "mean_dollars_nonzero": float(dollars[~zero_mask].mean()),
            "frac_supplement_zero": float(np.mean([rows[i]["ever_supplement"] for i in range(len(rows)) if zero_mask[i]])),
            "frac_supplement_nonzero": float(np.mean([rows[i]["ever_supplement"] for i in range(len(rows)) if not zero_mask[i]])),
            "mean_nfy_zero": float(n_fy[zero_mask].mean()),
            "mean_nfy_nonzero": float(n_fy[~zero_mask].mean()),
        }
        # logistic-ish: is zero-pub predicted by age? point-biserial
        rb, pb = stats.pointbiserialr(zero_mask.astype(float), age)
        report["zero_vs_nonzero"]["pointbiserial_zero_age"] = [float(rb), float(pb)]
        rbf, pbf = stats.pointbiserialr(zero_mask.astype(float), n_fy)
        report["zero_vs_nonzero"]["pointbiserial_zero_nfy"] = [float(rbf), float(pbf)]

    out_f = CACHE / "analysis.json"
    out_f.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# PROVENANCE: regenerate (do not hand-edit cache/analysis.json) with
#   uv run \
#       python -m artifacts_scripts.funding_spike.analyze
# Consumes cache/dataset.json (produced by collect.py). Writes cache/analysis.json.
# ---------------------------------------------------------------------------
