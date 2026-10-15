"""Each denoiser's paired difference from a reference denoiser, as a figure (Appendix B).

The intervals `denoiser_difference_cis` tabulates, drawn: one panel per column of
Table II (GIST and Gemma, target gap coverage and proposal gap percentile), one row
per denoiser, a dot at its mean per-topic difference from the reference and a bar
for the paired 95% t-interval, with zero (the reference itself) marked. A dot is
filled black where its interval excludes zero and hollow grey where it does not, so
what is significant reads in print and without colour; a legend under the panels
says which is which.

Paired rather than per-denoiser intervals because the topics vary far more than the
denoisers do: a mean's own interval would be wide enough to hide every difference
the pairing shows.

Writes pgfplots (needs \\usepgfplotslibrary{groupplots}) sized for a full-width
figure in a two-column page.

Usage (from the bookgap root)::

    uv run python -m artifacts_scripts.paper.denoiser_difference_plot --topics all \\
        --reference LPP --output paper/artifacts/denoiser_differences.tex RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from artifacts_scripts.common import friendly_denoiser_name, friendly_embedder_name, latex_provenance
from artifacts_scripts.paper.denoiser_difference_cis import TOPIC_SETS, Difference, Run, paired_difference, read_run
from artifacts_scripts.paper.regen_sweep import DENOISERS, EMBEDDERS

# Table II's columns, in its order: (embedder key, metric).
PANELS = (("gist", "gap_coverage_fraction"), ("gemma", "gap_coverage_fraction"),
          ("gist", "proposal_gap_size_percentile"), ("gemma", "proposal_gap_size_percentile"))
PANEL_METRIC = {"gap_coverage_fraction": "coverage", "proposal_gap_size_percentile": "percentile"}
# A dot, its bar and its legend image: filled black where the interval excludes zero.
SIGNIFICANT = "black, mark=*, only marks, mark size=1.8pt"
NOT_SIGNIFICANT = "gray, mark=o, mark options={fill=white}, only marks, mark size=1.8pt"
LEGEND = "denoiser-differences-legend"


def rows(runs: list[Run], reference: str) -> list[str]:
    """The denoisers drawn, top to bottom: every one but the reference, in the sweep's
    order (Table II's)."""
    present = {r.denoiser for r in runs}
    unknown = present - set(DENOISERS) - {reference}
    if unknown:
        raise SystemExit(f"error: denoisers outside the sweep's order: {sorted(unknown)}")
    return [d for d in DENOISERS if d in present and d != reference]


def differences(runs: list[Run], reference: str, embedder: str, metric: str,
                denoisers: list[str]) -> list[Difference]:
    model = EMBEDDERS[embedder][0]
    by_denoiser = {r.denoiser: r for r in runs if r.embedder == model}
    ref = by_denoiser[reference]
    return [paired_difference(by_denoiser[d].scores[metric], ref.scores[metric]) for d in denoisers]


def _coords(points: list[tuple[int, Difference]]) -> str:
    return " ".join(f"({d.mean:.2f},{y}) +- ({d.half_width:.2f},0)" for y, d in points)


def render(runs: list[Run], reference: str) -> str:
    denoisers = rows(runs, reference)
    labels = ",".join("{" + friendly_denoiser_name(d) + "}" for d in denoisers)
    n = len(denoisers)
    lines = [
        # font on the picture: a groupplot passes only pgfplots keys to its axes.
        r"\begin{tikzpicture}[font=\footnotesize]",
        r"\begin{groupplot}[",
        r"    group style={group size=4 by 1, horizontal sep=10pt, yticklabels at=edge left},",
        r"    scale only axis, width=0.2\textwidth, height=" + f"{13 * n}pt,",
        r"    title style={font=\footnotesize},",
        f"    ymin=0.4, ymax={n + 0.6:g}, y dir=reverse, ytick={{1,...,{n}}}, yticklabels={{{labels}}},",
        r"    tick align=outside, major tick length=2pt, ytick style={draw=none},",
        r"    xmajorgrids, grid style={gray!20},",
        r"    error bars/error mark=none, error bars/error bar style={line width=0.8pt},",
        r"    clip=false,",
        r"]",
    ]
    for panel, (embedder, metric) in enumerate(PANELS):
        diffs = differences(runs, reference, embedder, metric, denoisers)
        title = f"{friendly_embedder_name(EMBEDDERS[embedder][0])} {PANEL_METRIC[metric]}"
        # The legend is built in the first panel from images of its own, not from the
        # plotted points, so it holds both entries whichever panel has which kind.
        legend = (f", legend to name={LEGEND}, legend columns=2, legend style={{draw=none,"
                  r" /tikz/every odd column/.append style={column sep=2pt},"
                  r" /tikz/every even column/.append style={column sep=12pt}}" if panel == 0 else "")
        lines += [
            f"\\nextgroupplot[title={{{title}}}{legend}]",
            f"\\draw[gray!60] (axis cs:0,0.4) -- (axis cs:0,{n + 0.6:g});",
        ]
        if panel == 0:
            lines += [
                f"\\addlegendimage{{{SIGNIFICANT}}}",
                r"\addlegendentry{95\% CI excludes 0}",
                f"\\addlegendimage{{{NOT_SIGNIFICANT}}}",
                r"\addlegendentry{95\% CI includes 0}",
            ]
        significant = [(y, d) for y, d in enumerate(diffs, 1) if d.excludes_zero]
        not_significant = [(y, d) for y, d in enumerate(diffs, 1) if not d.excludes_zero]
        if not_significant:
            lines.append(f"\\addplot[{NOT_SIGNIFICANT}, error bars/.cd, x dir=both, x explicit] coordinates {{"
                         + _coords(not_significant) + "};")
        if significant:
            lines.append(f"\\addplot[{SIGNIFICANT}, error bars/.cd, x dir=both, x explicit] coordinates {{"
                         + _coords(significant) + "};")
    lines += [
        r"\end{groupplot}",
        r"\path (group c1r1.south west) -- (group c4r1.south east) node[midway, below=14pt] (xlabel)",
        f"    {{Difference from {friendly_denoiser_name(reference)} (points), with paired 95\\% CI}};",
        f"\\node[anchor=north, inner sep=0pt] at (xlabel.south) {{\\pgfplotslegendfromname{{{LEGEND}}}}};",
        r"\end{tikzpicture}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topics", choices=sorted(TOPIC_SETS), required=True,
                    help="compare over the fixed random 20 topics, or every topic the runs scored")
    ap.add_argument("--reference", required=True, help="the denoiser every other is compared with (e.g. LPP)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("run_dirs", type=Path, nargs="+")
    args = ap.parse_args()
    read = [read_run(d, TOPIC_SETS[args.topics]) for d in args.run_dirs]
    shared = {s for s, _ in read}
    if len(shared) != 1:
        print(f"error: the runs differ in prompt articles, books or radius fraction: {sorted(shared)}", file=sys.stderr)
        return 2
    runs = [r for _, r in read]
    keys = [(r.embedder, r.denoiser) for r in runs]
    if len(set(keys)) != len(keys):
        print(f"error: more than one run for an (embedder, denoiser): {sorted(k for k in keys if keys.count(k) > 1)}",
              file=sys.stderr)
        return 2
    args.output.write_text(render(runs, args.reference) + latex_provenance(__spec__.name))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
