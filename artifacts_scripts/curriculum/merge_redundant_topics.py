"""#2 — coherence-balance by merging keyword-redundant topics (the bottom-up dual of
"stop splitting a branch once it's one concept").

docs/backlog/curriculum-keyword-concept-seeding.md. Embedding clustering allocates topics ∝
density, so a dense literature (speech) over-splits into many near-duplicate chapters and a
generic blob (proceedings/paratext) forms several. Fix it AFTER clustering: compute each topic's
keyword distribution, merge topics whose Jensen–Shannon divergence is below τ.

Why JS (not Jaccard/Simpson): it's the canonical distributional *effect size* — density-invariant
(unlike a likelihood-ratio/G-test, whose statistic grows with N and would re-introduce the density
bias we're removing). Why COMPLETE-linkage (merge only if ALL pairwise JS<τ): single-linkage chains
through the paratext hub (its generic NLP terms bridge to LLMs/dialogue), merging distinct content;
complete-linkage requires mutual redundancy.

τ: ``--tau`` (sweep for the diagnostic), or ``--permutation-null`` to set it knob-free as the JS
between two random halves of the SAME topic (the null "same-concept" divergence) — merge any pair as
redundant as same-concept halves. Re-extracts merged-topic keywords via the cached field index.

Example:
  python -m artifacts_scripts.curriculum.merge_redundant_topics \\
      --run-dir artifacts/curriculum/jurafsky_trackA_dec --field 17 --tau 0.3 \\
      --out artifacts/curriculum/jurafsky_trackA_dec_merged
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from hydra.utils import instantiate
from omegaconf import OmegaConf

from artifacts_scripts.curriculum.config import CurriculumConfig
from artifacts_scripts.curriculum.field_term_index import build_field_term_index
from bookgap.util import LOGGER, setup_logging

_MODULE = "artifacts_scripts.curriculum.merge_redundant_topics"
_EPS = 0.5


def _js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))
    return max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0)


def _profiles_to_matrix(profiles):
    """list of {term: weight} -> row-normalized distribution matrix over the union vocab."""
    vocab = sorted({t for d in profiles for t in d})
    vi = {t: i for i, t in enumerate(vocab)}
    P = np.zeros((len(profiles), len(vocab)))
    for r, d in enumerate(profiles):
        for t, s in d.items():
            P[r, vi[t]] = max(float(s), 0.0)
    return P / np.maximum(P.sum(1, keepdims=True), 1e-12)


def _js_matrix(P):
    n = len(P); J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = J[j, i] = _js(P[i], P[j])
    return J


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path, help="has topic_keywords.json + topic_work_ids.json")
    ap.add_argument("--field", default="17")
    ap.add_argument("--tau", type=float, default=0.3, help="merge topics with complete-linkage JS < tau")
    ap.add_argument("--permutation-null", action="store_true",
                    help="set tau knob-free = 95th pctile JS between two random halves of the same topic")
    ap.add_argument("--top-k", type=int, default=30, help="keywords per merged topic")
    ap.add_argument("--min-df", type=int, default=20, help="field index cache key")
    ap.add_argument("--max-features", type=int, default=1_000_000, help="field index cache key")
    ap.add_argument("--snapshot-date", default="2025-11-06")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    setup_logging(log_name="merge_topics", log_level="INFO", include_time=False)

    kw = json.loads((args.run_dir / "topic_keywords.json").read_text())
    twi = json.loads((args.run_dir / "topic_work_ids.json").read_text())
    topics = sorted(kw.keys(), key=lambda x: int(x))
    profiles = [{k: float(s) for k, s in kw[t]} for t in topics]
    n = len(topics)

    # cached field index (for re-extracting merged-topic keywords + the permutation null)
    X, vocab, field_df = build_field_term_index(args.field, args.snapshot_date, args.min_df, args.max_features)
    cfg = OmegaConf.structured(CurriculumConfig()); cfg.experiment.cloud.filter_pair.pre.field = str(args.field)
    cloud = instantiate(cfg.experiment.cloud)
    wid = cloud.work_ids(); titles = cloud.work_titles(); id2idx = {w: i for i, w in enumerate(wid)}
    BG = X.shape[0]

    def w2k(rows, k):
        """W2K over a set of field rows -> {term: lift} top-k (lift vs the whole field)."""
        if not len(rows):
            return {}
        cnt = np.asarray(X[rows].sum(0)).ravel()
        nz = cnt.nonzero()[0]; nS = len(rows); out = {}
        for tid in nz:
            c = cnt[tid]
            lift = np.log(((c + _EPS) / (nS + _EPS)) / ((field_df[tid] + _EPS) / (BG + _EPS)))
            if lift > 0:
                out[int(tid)] = lift * np.log1p(c)
        return dict(sorted(out.items(), key=lambda x: -x[1])[:k])

    rows_of = {t: [id2idx[w] for w in twi.get(t, []) if w in id2idx] for t in topics}

    # τ via permutation null: JS between two random halves of each topic (same-concept divergence)
    if args.permutation_null:
        rng = np.random.default_rng(0); nulls = []
        for t in topics:
            r = rows_of[t]
            if len(r) < 8:
                continue
            r = list(r); rng.shuffle(r); h = len(r) // 2
            pa = w2k(r[:h], args.top_k); pb = w2k(r[h:], args.top_k)
            if pa and pb:
                Pn = _profiles_to_matrix([{vocab[i]: s for i, s in pa.items()},
                                          {vocab[i]: s for i, s in pb.items()}])
                nulls.append(_js(Pn[0], Pn[1]))
        tau = float(np.percentile(nulls, 95)) if nulls else args.tau
        LOGGER.info("permutation null: same-concept-half JS mean=%.3f, 95th=%.3f -> tau=%.3f",
                    float(np.mean(nulls)), tau, tau)
    else:
        tau = args.tau

    # complete-linkage agglomerative merge on the JS distances, cut at τ
    P = _profiles_to_matrix(profiles); J = _js_matrix(P)
    Z = linkage(squareform(J, checks=False), method="complete")
    lab = fcluster(Z, t=tau, criterion="distance")
    groups = defaultdict(list)
    for i, c in enumerate(lab):
        groups[c].append(i)
    order = sorted(groups.values(), key=lambda m: -sum(len(rows_of[topics[x]]) for x in m))
    LOGGER.info("merge (complete-linkage, tau=%.3f): %d -> %d topics", tau, n, len(order))

    # write the merged run-dir: union work ids + re-extracted keywords + summary
    args.out.mkdir(parents=True, exist_ok=True)
    out_twi = {}; out_kw = {}; sm = [f"# Merged curriculum (JS<{tau:.2f}) — {args.run_dir.name}", ""]
    for new_i, members in enumerate(order):
        ids = [w for x in members for w in twi.get(topics[x], [])]
        rows = [id2idx[w] for w in ids if w in id2idx]
        mk = w2k(rows, args.top_k)
        out_twi[str(new_i)] = ids
        out_kw[str(new_i)] = [[vocab[i], float(s)] for i, s in mk.items()]
        merged_note = (" [merged %d]" % len(members)) if len(members) > 1 else ""
        sm += [f"## Topic {new_i}{merged_note}  ({len(ids)} works)", "",
               ", ".join(f"`{vocab[i]}`" for i in list(mk)[:20]), ""]
        if len(members) > 1:
            sm.append("  ↳ from: " + " | ".join(
                ",".join(vocab[i] for i in list(w2k(rows_of[topics[x]], 4))[:4]) for x in members) + "\n")
    (args.out / "topic_work_ids.json").write_text(json.dumps(out_twi))
    (args.out / "topic_keywords.json").write_text(json.dumps(out_kw))
    cmd = f"python -m {_MODULE} " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    (args.out / "summary.md").write_text("\n".join(sm) + f"\n---\n*Generated by `{cmd}`*\n")
    print(f"merged {n} -> {len(order)} topics (tau={tau:.3f}) -> {args.out}")
    for members in order:
        if len(members) > 1:
            print("  merged: " + " | ".join(",".join(sorted(profiles[x], key=lambda k:-profiles[x][k])[:3]) for x in members))


if __name__ == "__main__":
    main()
