"""Generate a per-proposal appendix .tex from a gap-analysis run output dir.

Emits one page per proposal in the run's ``gaps_analysis_results.csv``,
containing:
- topic name, topic id, summary stats
- proposed textbook title + abstract
- the 10 prompt articles (titles + abstracts), in gap-centrality order
- the highest-coverage existing book in the topic (title + abstract +
  coverage stats), or a note if the topic has no books

Reads ``prompts/gap_<gi>_topic_<topic_id>.json`` for the actual prompt
articles given to the LLM, and ``contexts/gap_<gi>_topic_<topic_id>.json``
for the highest-coverage existing book. Fails fast if either directory is
absent — runs that predate the JSON artifact stack need to be re-run via
the gap-analysis pipeline.

Usage::

    uv run python -m artifacts_scripts.paper.generate_proposal_appendix \\
        --run-dir outputs/gemma4_sweep/2026-05-22_11-11-43_gist_a10_b5_dropk10 \\
        --output  paper/artifacts/proposal_appendix.tex
"""
from __future__ import annotations

import argparse
import math
from typing import Literal

BookMarkerStyle = Literal["physics", "linear"]
_DEFAULT_BOOK_MARKER_STYLE: BookMarkerStyle = "physics"

# Default curated appendix: the six topics named as failures in §5.1
# (\Cref{sec:overall_performance}) and T10181 (Natural Language Processing
# Techniques) as a representative successful case. T10126 and T12026
# (parse-recovered via lenient parsing — see
# artifacts_scripts/lenient_recover_failed_proposals.py) are commented out
# for the current paper revision; uncomment to include them in the appendix.
DEFAULT_TOPIC_IDS = (
    # "T10126",
    "T10181",
    "T10215",
    "T10639",
    # "T12026",
    "T12157",
    "T13623",
    "T13674",
    "T14381",
)
import csv
import html
import re
import unicodedata
from pathlib import Path

from artifacts_scripts.common import latex_provenance
from bookgap.llm.prompt_record import (
    ArticleEntry,
    BookCoverage,
    BookEntry,
    LocalProjection2D,
    ProposalCoverage,
    read_context_record,
    read_prompt_record,
    record_filename,
)

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\n\r\f\v]+")

# ASCII bell as an internal sentinel for "unsupported character" — survives
# NFKC normalization, the whitespace collapser, and latex_escape unchanged.
# At LaTeX-render time it becomes the `\unkchar` macro defined at the top
# of the generated file.
_UNK_SENTINEL = "\x07"
_UNK_LATEX = r"\unkchar{}"

def _safe_for_pdflatex(ch: str) -> str:
    """Map characters pdflatex can't render to an unsupported-glyph sentinel.

    Keep ASCII printable, plus accented Latin letters (Latin-1 letters
    0xC0-0xFF, Latin Extended-A 0x100-0x17F). Replace everything else
    (Greek, CJK, Cyrillic, Arabic, math-symbol blocks, fullwidth chars,
    Latin-1 symbols like guillemets and degree-sign) with the sentinel,
    which renders as a visible `\\unkchar` tofu block.
    """
    if ch == _UNK_SENTINEL:
        return ch
    cp = ord(ch)
    if cp < 0x20:
        return " " if ch in "\t\n\r" else ""
    if cp < 0x7F:
        return ch
    if 0xC0 <= cp <= 0x017F:
        return ch
    return _UNK_SENTINEL


def clean_text(text: str) -> str:
    """Strip HTML/MathML tags, decode entities, normalize, collapse whitespace."""
    if not text:
        return ""
    s = _HTML_TAG_RE.sub(" ", text)
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    s = "".join(_safe_for_pdflatex(c) for c in s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def latex_escape(text: str) -> str:
    if not text:
        return ""
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in text)


def tex(text: str, max_chars: int | None = None) -> str:
    s = clean_text(text)
    if max_chars is not None and len(s) > max_chars:
        cut = s.rfind(" ", 0, max_chars)
        if cut < max_chars // 2:  # no good boundary — hard cut
            cut = max_chars
        s = s[:cut].rstrip(" ,.;:") + "..."
    return latex_escape(s).replace(_UNK_SENTINEL, _UNK_LATEX)


_PROMPT_TITLE_MAX = 75
_PROMPT_ABSTRACT_MAX = 0

_OPENALEX_ID_RE = re.compile(r"https?://openalex\.org/(?:works/|topics/)?([A-Z]\d+)")


def _openalex_short(url: str) -> str:
    """Extract the W12345 / T12345 suffix from an OpenAlex URL."""
    m = _OPENALEX_ID_RE.search(url) if url else None
    return m.group(1) if m else (url or "")


def _openalex_href(url: str) -> str:
    """``\\href{url}{\\texttt{W12345}}`` — empty if url is falsy."""
    if not url:
        return ""
    return f"\\href{{{url}}}{{\\texttt{{{_openalex_short(url)}}}}}"


def _gray(text: str) -> str:
    """Subdued paragraph-label style."""
    return f"\\textcolor{{gray}}{{{text}}}"


_FIGURE_HEIGHT_CM = 6.0       # shared height of both side-by-side figures
_STACK_BAR_WIDTH_CM = 0.5     # width of each stack column
_STACK_GAP_CM = 0.15          # horizontal gap between pre and post stacks

# Book-marker physics: existing books are rendered at a "perceived
# distance" derived from their coverage of the topic's article cloud.
# Higher-coverage books are drawn closer-and-brighter; lower-coverage
# ones recede into atmospheric haze. Two physics relationships:
#   apparent size      ∝ 1 / d           (small-angle approximation)
#   apparent intensity ∝ exp(-α · d)     (Beer-Lambert haze)
# We map coverage c ∈ [0, c_max] linearly to d ∈ [_BOOK_D_MIN, _BOOK_D_MAX]
# (closest at c=c_max, farthest at c=0). _BOOK_BASE_SIZE_PT is the
# closest book's font size; _BOOK_HAZE_ALPHA controls how much the
# farthest book fades.
# Base font for scatter markers: comfortably available in CM, so no
# substitution; per-marker size comes from scale= instead.
_MARKER_BASE_PT = 8.0
_BOOK_D_MIN = 1.0
_BOOK_D_MAX = 2.5  # size ratio is _BOOK_D_MAX / _BOOK_D_MIN
_BOOK_BASE_SIZE_PT = 12.0  # 1.5× larger than the original 4pt baseline
_BOOK_HAZE_ALPHA = 0.4


def _render_twin_stacks_tikz(
    pre_counts: list[int],
    post_counts: list[int],
    proposal_post_count: int,
) -> str:
    """Twin stacks (pre on the left, post on the right) of equal total
    height — articles redistribute when the proposal is added as a book,
    they aren't created or destroyed. Each stack is a single column with
    one segment per existing book, sorted by pre-coverage desc; segments
    alternate between dark and light blue so adjacent books are visually
    separable. The post stack tops with a green slab for the proposal's
    net absorption.

    Total figure height is fixed to ``_FIGURE_HEIGHT_CM`` (matching the
    scatter helper) via an explicit bounding box, so the two figures
    align cleanly side-by-side.
    """
    total = max(sum(pre_counts), sum(post_counts) + proposal_post_count, 1)
    yscale = _FIGURE_HEIGHT_CM / total
    x_right = _STACK_BAR_WIDTH_CM + _STACK_GAP_CM
    total_width = 2 * _STACK_BAR_WIDTH_CM + _STACK_GAP_CM

    out: list[str] = []
    out.append("\\begin{tikzpicture}")
    out.append(
        f"\\useasboundingbox (0,0) rectangle "
        f"({total_width:.3f}cm,{_FIGURE_HEIGHT_CM:.3f}cm);"
    )
    if not pre_counts and proposal_post_count == 0:
        out.append("\\end{tikzpicture}")
        return "\n".join(out)
    # Left stack: pre counts.
    bottom = 0
    for i, c in enumerate(pre_counts):
        if c <= 0:
            continue
        color = "blue!70" if i % 2 == 0 else "blue!35"
        out.append(
            f"\\fill[{color}] (0cm,{bottom * yscale:.4f}cm) rectangle "
            f"({_STACK_BAR_WIDTH_CM:.3f}cm,{(bottom + c) * yscale:.4f}cm);"
        )
        bottom += c
    # Right stack: post counts (same alternating shades, same book order).
    bottom = 0
    for i, c in enumerate(post_counts):
        if c <= 0:
            continue
        color = "blue!70" if i % 2 == 0 else "blue!35"
        out.append(
            f"\\fill[{color}] ({x_right:.3f}cm,{bottom * yscale:.4f}cm) "
            f"rectangle ({x_right + _STACK_BAR_WIDTH_CM:.3f}cm,"
            f"{(bottom + c) * yscale:.4f}cm);"
        )
        bottom += c
    # Proposal slab on top of right stack.
    if proposal_post_count > 0:
        out.append(
            f"\\fill[green!60!black] ({x_right:.3f}cm,{bottom * yscale:.4f}cm) "
            f"rectangle ({x_right + _STACK_BAR_WIDTH_CM:.3f}cm,"
            f"{(bottom + proposal_post_count) * yscale:.4f}cm);"
        )
    out.append("\\end{tikzpicture}")
    return "\n".join(out)


def _render_projection_scatter_tikz(
    projection: LocalProjection2D,
    book_coverages: list[int],
    book_marker_style: BookMarkerStyle = _DEFAULT_BOOK_MARKER_STYLE,
) -> str:
    """TikZ scatter of gap articles + books + proposal in PCA-on-all-articles
    coordinates. ``book_coverages`` is parallel to ``projection.books`` and
    provides each book's in-gap article count, used to set per-book opacity
    (more covered articles → more opaque). The seed marker is omitted —
    seeds are a gap-detection provenance artifact, not visually informative.

    ``book_marker_style`` selects how coverage maps to marker appearance:

    - ``"physics"`` (default): coverage maps to a perceived distance
      ``d``; size is ``∝ 1/d`` (small-angle) and opacity is
      ``∝ exp(-α·d)`` (Beer-Lambert haze). Less-covered books shrink
      *and* fade, evoking a depth cue. Constants are
      ``_BOOK_D_MIN/MAX``, ``_BOOK_BASE_SIZE_PT``, ``_BOOK_HAZE_ALPHA``.
    - ``"linear"``: original styling — constant scriptsize × glyph,
      opacity linearly interpolated from 0.15 (least covered) to 1.0
      (most covered).
    """
    all_xs = (
        [p.x for p in projection.gap_articles]
        + [p.x for p in projection.books]
        + [projection.proposal.x]
    )
    all_ys = (
        [p.y for p in projection.gap_articles]
        + [p.y for p in projection.books]
        + [projection.proposal.y]
    )
    extent = max(
        abs(min(all_xs)), abs(max(all_xs)),
        abs(min(all_ys)), abs(max(all_ys)),
        1e-6,
    )
    # Square plot of side _FIGURE_HEIGHT_CM, centered at the origin.
    half_side_cm = _FIGURE_HEIGHT_CM / 2
    scale = half_side_cm / extent  # cm per ND unit
    # Legend rendered below the plot in two rows; reserve vertical space
    # in the bounding box so the surrounding LaTeX flow accounts for it.
    legend_band_cm = 1.0

    out: list[str] = []
    out.append("\\begin{tikzpicture}")
    out.append(
        f"\\useasboundingbox ({-half_side_cm:.3f}cm,"
        f"{-half_side_cm - legend_band_cm:.3f}cm) "
        f"rectangle ({half_side_cm:.3f}cm,{half_side_cm:.3f}cm);"
    )
    # Books FIRST so articles render on top (a dense gap-article cloud
    # is the main thing; books should peek through where they overlap).
    # Per-book opacity: scale from 0.15 (no in-gap coverage) up to 1.0
    # (the topic's top-covering book). Books that cover nothing are
    # nearly invisible — exactly the visual weight they deserve.
    if projection.books:
        max_count = max(book_coverages) if book_coverages else 0
        for pt, count in zip(projection.books, book_coverages, strict=True):
            if book_marker_style == "physics":
                if max_count > 0:
                    # Linear map: c=c_max → d=D_MIN (closest);
                    # c=0 → d=D_MAX (farthest).
                    d = _BOOK_D_MIN + (_BOOK_D_MAX - _BOOK_D_MIN) * (
                        1.0 - count / max_count
                    )
                else:
                    d = _BOOK_D_MAX
                opacity = math.exp(-_BOOK_HAZE_ALPHA * (d - _BOOK_D_MIN))
                size_pt = _BOOK_BASE_SIZE_PT * _BOOK_D_MIN / d
                # Size via a TikZ transform, not \fontsize. Requesting a font
                # smaller than the family provides makes TeX silently substitute
                # the nearest available size -- Computer Modern bottoms out around
                # 5pt, so every marker below that rendered identically and the
                # size-encodes-coverage channel was clipped flat at the low end
                # (396 substitution warnings, smallest request 2.4pt). scale= is a
                # geometric transform, so it is continuous all the way down.
                font_spec = (
                    f"font=\\fontsize{{{_MARKER_BASE_PT:g}pt}}{{0pt}}\\selectfont, "
                    f"scale={size_pt / _MARKER_BASE_PT:.3f}"
                )
            elif book_marker_style == "linear":
                opacity = (
                    0.15 + 0.85 * (count / max_count) if max_count > 0 else 0.15
                )
                font_spec = "font=\\scriptsize"
            else:
                raise ValueError(
                    f"unknown book_marker_style={book_marker_style!r}"
                )
            out.append(
                f"\\node[blue, opacity={opacity:.3f}, {font_spec}] at "
                f"({pt.x * scale:.4f}cm,{pt.y * scale:.4f}cm) {{$\\times$}};"
            )
    # Gap articles: black for those still covered by an existing book,
    # green for those that flip to the proposal's Voronoi cell.
    for pt, flipped in zip(
        projection.gap_articles,
        projection.gap_articles_flipped_to_proposal,
        strict=True,
    ):
        color = "green!60!black" if flipped else "black"
        out.append(
            f"\\fill[{color}] ({pt.x * scale:.4f}cm,{pt.y * scale:.4f}cm) "
            f"circle (0.5pt);"
        )
    # Proposal on top. Drawn as a TikZ `star` shape so we get a black
    # border around the green fill; this requires
    # \usetikzlibrary{shapes.geometric} (loaded by main.tex).
    out.append(
        f"\\node[star, star points=5, star point ratio=2.25, "
        f"draw=black, line width=0.4pt, fill=green!60!black, "
        f"minimum size=9pt, inner sep=0pt] at "
        f"({projection.proposal.x * scale:.4f}cm,"
        f"{projection.proposal.y * scale:.4f}cm) {{}};"
    )
    # Legend: two rows of two entries each, below the plot. Marker
    # styles mirror those used inside the plot.
    legend_y1 = -half_side_cm - 0.35
    legend_y2 = -half_side_cm - 0.75
    col_left_marker = -half_side_cm + 0.20
    col_left_text = -half_side_cm + 0.45
    col_right_marker = 0.20
    col_right_text = 0.45
    # Each entry: (marker_x, text_x, y, marker_tikz, label).
    # The proposal entry uses the bordered TikZ star shape to mirror the
    # in-plot marker; the others are inline math glyphs in scriptsize.
    legend_entries = [
        (col_left_marker, col_left_text, legend_y1,
         "\\node[blue, font=\\scriptsize] at "
         f"({col_left_marker:.3f}cm,{legend_y1:.3f}cm) {{$\\times$}};",
         "existing books"),
        (col_left_marker, col_left_text, legend_y2,
         "\\node[black, font=\\scriptsize] at "
         f"({col_left_marker:.3f}cm,{legend_y2:.3f}cm) {{$\\bullet$}};",
         "other in-gap articles"),
        (col_right_marker, col_right_text, legend_y1,
         "\\node[green!60!black, font=\\scriptsize] at "
         f"({col_right_marker:.3f}cm,{legend_y1:.3f}cm) {{$\\bullet$}};",
         "in-gap articles flipped to proposal"),
        (col_right_marker, col_right_text, legend_y2,
         "\\node[star, star points=5, star point ratio=2.25, "
         "draw=black, line width=0.3pt, fill=green!60!black, "
         "minimum size=6pt, inner sep=0pt] at "
         f"({col_right_marker:.3f}cm,{legend_y2:.3f}cm) {{}};",
         "proposal"),
    ]
    for _mx, tx, ly, marker_node, label in legend_entries:
        out.append(marker_node)
        out.append(
            f"\\node[anchor=west, font=\\scriptsize] at "
            f"({tx:.3f}cm,{ly:.3f}cm) {{{label}}};"
        )
    out.append("\\end{tikzpicture}")
    return "\n".join(out)


def emit_topic_page(
    *,
    topic_id: str,
    topic_name: str,
    prop_title: str,
    prop_abstract: str,
    gap_coverage: float,
    gap_size_pct: float | None,
    num_books: int,
    num_articles: int,
    articles: list[ArticleEntry],
    prompt_books: list[BookEntry],
    proposal_coverage: ProposalCoverage,
    covering_books_for_chart: list[BookCoverage],
    local_projection: LocalProjection2D,
    book_marker_style: BookMarkerStyle = _DEFAULT_BOOK_MARKER_STYLE,
) -> str:
    out: list[str] = []
    out.append("\\clearpage")
    out.append("\\begingroup\\small")
    topic_url = f"https://openalex.org/topics/{topic_id}"
    topic_link = (
        f"\\href{{{topic_url}}}{{\\texttt{{{latex_escape(topic_id)}}}}}"
    )
    # ID before the name, both in the subsection title and the TOC entry.
    out.append(
        f"\\subsection[{latex_escape(topic_id)} {tex(topic_name)}]"
        f"{{{topic_link} {tex(topic_name)}}}"
    )
    out.append(f"\\label{{appendix:proposal-{latex_escape(topic_id)}}}")
    pct_str = (
        f"{gap_size_pct * 100:.1f}\\%" if gap_size_pct is not None else "n/a"
    )
    out.append(
        f"\\noindent\\textit{{Topic has {num_books} books and "
        f"{num_articles} articles; proposal covers "
        f"{gap_coverage * 100:.1f}\\% of the target gap; "
        f"target gap size percentile = {pct_str}.}}"
    )
    out.append("\\par\\medskip")

    scatter_tikz = _render_projection_scatter_tikz(
        local_projection,
        book_coverages=[b.articles_covered for b in covering_books_for_chart],
        book_marker_style=book_marker_style,
    )
    out.append("\\begin{center}")
    out.append(scatter_tikz)
    out.append("\\end{center}")
    out.append("\\par\\medskip")

    out.append(f"\\paragraph{{{_gray('Book proposal')}}}")
    out.append(f"\\textbf{{{tex(prop_title)}}}")
    out.append("")
    out.append(tex(prop_abstract))

    out.append(f"\\paragraph{{{_gray('Prompt articles')}}}")
    out.append("\\begin{enumerate}\\setlength\\itemsep{0.25em}")
    for a in articles:
        link = _openalex_href(a.openalex_id)
        id_prefix = f"{link} " if link else ""
        out.append(
            f"\\item {id_prefix}\\textbf{{\\textsf{{{tex(a.display_name, _PROMPT_TITLE_MAX)}}}}}"
        )
        if _PROMPT_ABSTRACT_MAX > 0:
            out.append("")
            out.append(tex(a.abstract, _PROMPT_ABSTRACT_MAX))
    out.append("\\end{enumerate}")

    out.append(f"\\paragraph{{{_gray('Prompt books')}}}")
    if not prompt_books:
        out.append("\\textit{No books were in the prompt.}")
    else:
        out.append("\\begin{enumerate}\\setlength\\itemsep{0.25em}")
        for b in prompt_books:
            link = _openalex_href(b.openalex_id)
            id_prefix = f"{link} " if link else ""
            out.append(
                f"\\item {id_prefix}\\textbf{{\\textsf{{{tex(b.display_name, _PROMPT_TITLE_MAX)}}}}}"
            )
            if _PROMPT_ABSTRACT_MAX > 0:
                out.append("")
                out.append(tex(b.abstract, _PROMPT_ABSTRACT_MAX))
        out.append("\\end{enumerate}")

    out.append("\\endgroup")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--sort",
        choices=("name", "coverage", "csv"),
        default="name",
        help="name=alphabetical by topic name; coverage=ascending by gap "
        "coverage fraction (worst first, to foreground data-quality "
        "cases); csv=original CSV order.",
    )
    ap.add_argument(
        "--topic-ids",
        nargs="+",
        default=list(DEFAULT_TOPIC_IDS),
        help="Topic IDs to emit pages for, in --sort order. Defaults to the "
        "curated nine that highlight the §5.1 failure cases, the two "
        "label-noise / parse-recovered topics, and one representative "
        "success (T10181).",
    )
    ap.add_argument(
        "--book-marker-style",
        choices=("physics", "linear"),
        default=_DEFAULT_BOOK_MARKER_STYLE,
        help="How existing-book markers in the per-topic scatter plot scale "
        "with coverage. 'physics' (default): size ∝ 1/d and opacity ∝ "
        "exp(-α·d) with d derived linearly from coverage — books shrink "
        "and fade into haze as coverage drops. 'linear': original styling "
        "with constant marker size and opacity linearly interpolated from "
        "0.15 (least covered) to 1.0 (most covered).",
    )
    args = ap.parse_args()

    csv_path = args.run_dir / "gaps_analysis_results.csv"
    prompts_dir = args.run_dir / "prompts"
    contexts_dir = args.run_dir / "contexts"
    for d in (prompts_dir, contexts_dir):
        if not d.is_dir():
            raise SystemExit(
                f"missing required directory {d} -- this run predates the JSON "
                f"artifact stack; re-run the gap-analysis pipeline to produce "
                f"prompts/ and contexts/"
            )

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    if args.topic_ids:
        wanted = set(args.topic_ids)
        rows = [r for r in rows if r["topic_id"] in wanted]
        missing = wanted - {r["topic_id"] for r in rows}
        if missing:
            raise SystemExit(
                f"--topic-ids included unknown id(s): {sorted(missing)}"
            )

    if args.sort == "name":
        rows.sort(key=lambda r: r["topic_name"].lower())
    elif args.sort == "coverage":
        rows.sort(key=lambda r: float(r["gap_coverage_fraction"]))

    chunks: list[str] = []
    for row in rows:
        topic_id = row["topic_id"]
        gap_idx = 0  # headline runs are single-gap (gap_idx=0)
        prompt_path = prompts_dir / record_filename(gap_idx, topic_id)
        ctx_path = contexts_dir / record_filename(gap_idx, topic_id)
        if not prompt_path.is_file() or not ctx_path.is_file():
            raise SystemExit(
                f"missing JSON for {topic_id} "
                f"(expected {prompt_path} and {ctx_path})"
            )
        prompt = read_prompt_record(prompt_path)
        if prompt.gap_idx != gap_idx:
            raise SystemExit(
                f"prompt record for {topic_id} has gap_idx={prompt.gap_idx}, "
                f"expected {gap_idx}"
            )
        context = read_context_record(ctx_path)

        chunks.append(
            emit_topic_page(
                topic_id=topic_id,
                topic_name=row["topic_name"],
                prop_title=row["proposal_title"],
                prop_abstract=row["proposal_abstract"],
                gap_coverage=float(row["gap_coverage_fraction"]),
                gap_size_pct=(
                    float(row["gap_size_percentile"])
                    if row["gap_size_percentile"]
                    else None
                ),
                num_books=int(row["num_books"]),
                num_articles=int(row["num_articles"]),
                articles=prompt.sampled_articles,
                prompt_books=prompt.covering_books,
                proposal_coverage=context.proposal_coverage,
                covering_books_for_chart=context.covering_books,
                local_projection=context.local_projection_2d,
                book_marker_style=args.book_marker_style,
            )
        )

    preamble = (
        "% Auto-generated by artifacts_scripts/generate_proposal_appendix.py "
        "-- do not edit by hand.\n"
        "\\providecommand{\\unkchar}{\\rule[0.1ex]{0.55em}{0.55em}}\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        preamble + "\n" + "\n\n".join(chunks) + "\n" + latex_provenance(__spec__.name)
    )
    print(f"Wrote {len(chunks)} topic pages to {args.output}")


if __name__ == "__main__":
    main()
