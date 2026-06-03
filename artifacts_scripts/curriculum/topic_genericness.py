"""Per-topic label-GENERICNESS = mean field-IDF of a topic's keywords (embedder-free).

docs/backlog/curriculum-keyword-concept-seeding.md. The "miscellaneous recent NLP" cluster (diverse
papers sharing only generic framing — `natural_language_processing/tasks/papers`) is invisible to
embedding cohesion (the embedder groups it as coherent) but obvious in keyword statistics: its
distinctive keywords are field-GENERIC (high document frequency → low IDF), whereas a content topic's
are field-SPECIFIC (`wer`, `word2vec` → low DF → high IDF). The W2K *lift* hides this (the misc cluster
concentrates the generic terms → high lift), so we read the raw field-IDF directly.

genericness(topic) = mean over its top-k keywords of IDF_field(t) = log(N_field / (DF_field(t)+1)).
Low = generic/miscellaneous label; high = specific content label. Reuses the cached field index; no
embedder. Doubles as the eval metric for keyword-TF-IDF clustering (does it raise mean specificity?).

Example:
  python -m artifacts_scripts.curriculum.topic_genericness \\
      --run-dir artifacts/curriculum/jurafsky_trackA_dec --field 17 --top-k 15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from artifacts_scripts.curriculum.field_term_index import build_field_term_index
from bookgap.util import setup_logging


def topic_idf(topic_keywords: dict, vocab, field_df, n_field: int, top_k: int = 15):
    """Return {topic_id: (mean_idf, [(kw, idf), ...])} over each topic's top-k keywords."""
    term2id = {t: i for i, t in enumerate(vocab)}
    idf = np.log(n_field / (field_df + 1.0))
    median_idf = float(np.median(idf))
    out = {}
    for t, entries in topic_keywords.items():
        kws = [k for k, _ in entries][:top_k]
        per = [(k, float(idf[term2id[k]]) if k in term2id else median_idf) for k in kws]
        vals = [v for _, v in per]
        out[t] = (float(np.mean(vals)) if vals else 0.0, per)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--field", default="17")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--min-df", type=int, default=20)
    ap.add_argument("--max-features", type=int, default=1_000_000)
    ap.add_argument("--snapshot-date", default="2025-11-06")
    args = ap.parse_args()
    setup_logging(log_name="genericness", log_level="WARNING", include_time=False)

    X, vocab, field_df = build_field_term_index(args.field, args.snapshot_date, args.min_df, args.max_features)
    kw = json.loads((args.run_dir / "topic_keywords.json").read_text())
    res = topic_idf(kw, vocab, field_df, X.shape[0], args.top_k)
    ranked = sorted(res.items(), key=lambda x: x[1][0])
    print(f"# topic genericness (mean field-IDF of top-{args.top_k} keywords) — {args.run_dir.name}")
    print(f"# low IDF = generic/miscellaneous label; high = specific.  (corpus median IDF "
          f"{float(np.median(np.log(X.shape[0] / (field_df + 1.0)))):.2f})\n")
    for t, (m, per) in ranked:
        tag = "  <- GENERIC" if m == ranked[0][1][0] or m < ranked[len(ranked)//2][1][0] * 0.7 else ""
        top = ", ".join(k for k, _ in per[:6])
        print(f"  topic {t:>2}  mean_idf={m:5.2f}  [{top}]{tag}")
    ms = [m for _, (m, _) in res.items()]
    print(f"\nmean across topics={np.mean(ms):.2f}  min={np.min(ms):.2f}  spread(max-min)={np.ptp(ms):.2f}")


if __name__ == "__main__":
    main()
