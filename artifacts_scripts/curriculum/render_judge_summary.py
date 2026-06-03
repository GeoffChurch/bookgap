"""Render a run-dir's topic_keywords + topic_work_ids into the judge-parseable summary.md.

The pairwise LM judge (eval_book_judge._parse_chapters_from_summary) expects per-topic blocks with a
``### Top Keywords`` `| Keyword | Score |` table and a ``### Top Works`` list of ``#### k. <title>``.
The keyword-clustering scripts (cluster_concept_cca, cluster_concept_by_keywords, merge_redundant_topics)
write a terser summary; this converts them so they can be judged. Overwrites summary.md in place
(keeps a summary.raw.md backup).

Example:
  python -m artifacts_scripts.curriculum.render_judge_summary --run-dir artifacts/curriculum/jurafsky_trackA_cca --field 17
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from artifacts_scripts.curriculum.config import CurriculumConfig
from bookgap.util import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--field", default="17")
    ap.add_argument("--n-keywords", type=int, default=30)
    ap.add_argument("--n-works", type=int, default=5)
    args = ap.parse_args()
    setup_logging(log_name="render_judge", log_level="WARNING", include_time=False)

    cfg = OmegaConf.structured(CurriculumConfig()); cfg.experiment.cloud.filter_pair.pre.field = str(args.field)
    cloud = instantiate(cfg.experiment.cloud)
    wid = cloud.work_ids(); titles = cloud.work_titles(); id2idx = {w: i for i, w in enumerate(wid)}

    kw = json.loads((args.run_dir / "topic_keywords.json").read_text())
    twi = json.loads((args.run_dir / "topic_work_ids.json").read_text())
    sp = args.run_dir / "summary.md"
    if sp.exists() and not (args.run_dir / "summary.raw.md").exists():
        (args.run_dir / "summary.raw.md").write_text(sp.read_text())

    lines = ["# Curriculum Summary", ""]
    for t in sorted(kw.keys(), key=lambda x: int(x)):
        lines += [f"## Topic {int(t) + 1}", "", "### Top Keywords", "", "| Keyword | Score |", "| :--- | :--- |"]
        for k, s in kw[t][:args.n_keywords]:
            lines.append(f"| {k} | {float(s):.4f} |")
        lines += ["", "### Top Works", ""]
        for j, w in enumerate(twi.get(t, [])[:args.n_works], 1):
            if w in id2idx:
                lines += [f"#### {j}. {titles[id2idx[w]]}", ""]
        lines.append("")
    sp.write_text("\n".join(lines))
    print(f"rendered judge-parseable summary.md ({len(kw)} topics) -> {sp}")


if __name__ == "__main__":
    main()
