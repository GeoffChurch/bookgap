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

    .conda/bin/python -m artifacts_scripts.generate_proposal_appendix \\
        --run-dir outputs/gemma4_sweep/2026-05-22_11-11-43_gist_a10_b5_dropk10 \\
        --output  paper/proposal_appendix.tex
"""
from __future__ import annotations

import argparse
import csv
import html
import re
import unicodedata
from pathlib import Path

from bookgap.llm.prompt_record import (
    ArticleEntry,
    BookCoverage,
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
_BOOK_ABSTRACT_MAX = 600  # highest-coverage book abstract (one per topic)

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


_BAR_CHART_MAX_WIDTH_EM = 16.0
_BAR_HEIGHT_EM = 0.35
_BAR_VSEP_EM = 0.4


def _render_bar_chart_tikz(
    book_labels: list[str],   # short labels (e.g. shortened titles)
    pre_counts: list[int],    # parallel
    post_counts: list[int],   # parallel
    proposal_post_count: int,
) -> str:
    """Horizontal bar chart: existing books (with overlaid pre/post bars)
    sorted by pre desc, then the proposal at the end (pre=0).
    """
    rows: list[tuple[str, int, int]] = list(
        zip(book_labels, pre_counts, post_counts, strict=True)
    )
    rows.append(("\\textit{proposal}", 0, proposal_post_count))

    max_count = max(
        [pre for _, pre, _ in rows] + [post for _, _, post in rows] + [1]
    )
    xscale = _BAR_CHART_MAX_WIDTH_EM / max_count

    out: list[str] = []
    out.append("\\begin{tikzpicture}[every node/.style={font=\\scriptsize}]")
    for i, (label, pre, post) in enumerate(rows):
        y = -i * (_BAR_HEIGHT_EM + _BAR_VSEP_EM)
        pre_w = pre * xscale
        post_w = post * xscale
        # Pre bar: pale gray.
        if pre > 0:
            out.append(
                f"\\fill[gray!25] (0em,{y:.3f}em) rectangle "
                f"({pre_w:.3f}em,{y + _BAR_HEIGHT_EM:.3f}em);"
            )
        # Post bar overlay: blue for existing books, green for proposal.
        color = "green!60!black" if label == "\\textit{proposal}" else "blue!50"
        if post > 0:
            out.append(
                f"\\fill[{color}] (0em,{y:.3f}em) rectangle "
                f"({post_w:.3f}em,{y + _BAR_HEIGHT_EM:.3f}em);"
            )
        # Label to the left, count to the right.
        out.append(
            f"\\node[anchor=east] at (-0.5em,{y + _BAR_HEIGHT_EM / 2:.3f}em) "
            f"{{{label}}};"
        )
        right_text = (
            f"{post}"
            if pre == 0
            else f"{pre} {{\\color{{gray}}\\textrightarrow}} {post}"
        )
        right_x = max(pre_w, post_w) + 0.3
        out.append(
            f"\\node[anchor=west] at ({right_x:.3f}em,{y + _BAR_HEIGHT_EM / 2:.3f}em) "
            f"{{{right_text}}};"
        )
    out.append("\\end{tikzpicture}")
    return "\n".join(out)


_SCATTER_HALF_SIDE_CM = 3.5  # half of the plot box width in cm


def _render_projection_scatter_tikz(
    projection: LocalProjection2D,
    book_coverages: list[int],
) -> str:
    """TikZ scatter of gap articles + books + proposal in PCA-on-all-articles
    coordinates. ``book_coverages`` is parallel to ``projection.books`` and
    provides each book's in-gap article count, used to set per-book opacity
    (more covered articles → more opaque). The seed marker is omitted —
    seeds are a gap-detection provenance artifact, not visually informative.
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
    scale = _SCATTER_HALF_SIDE_CM / extent  # cm per unit

    out: list[str] = []
    out.append("\\begin{tikzpicture}")
    # Books FIRST so articles render on top (a dense gap-article cloud
    # is the main thing; books should peek through where they overlap).
    # Per-book opacity: scale from 0.15 (no in-gap coverage) up to 1.0
    # (the topic's top-covering book). Books that cover nothing are
    # nearly invisible — exactly the visual weight they deserve.
    if projection.books:
        max_count = max(book_coverages) if book_coverages else 0
        for pt, count in zip(projection.books, book_coverages, strict=True):
            alpha = 0.15 + 0.85 * (count / max_count) if max_count > 0 else 0.15
            out.append(
                f"\\node[blue, opacity={alpha:.3f}, font=\\scriptsize] at "
                f"({pt.x * scale:.4f}cm,{pt.y * scale:.4f}cm) {{$\\times$}};"
            )
    # Gap articles: subdued grey for those still covered by an existing
    # book, green for those that flip to the proposal's Voronoi cell.
    for pt, flipped in zip(
        projection.gap_articles,
        projection.gap_articles_flipped_to_proposal,
        strict=True,
    ):
        color = "green!60!black" if flipped else "gray!60"
        out.append(
            f"\\fill[{color}] ({pt.x * scale:.4f}cm,{pt.y * scale:.4f}cm) "
            f"circle (0.5pt);"
        )
    # Proposal on top.
    out.append(
        f"\\node[green!60!black, font=\\bfseries] at "
        f"({projection.proposal.x * scale:.4f}cm,"
        f"{projection.proposal.y * scale:.4f}cm) {{$\\star$}};"
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
    top_book: BookCoverage | None,
    proposal_coverage: ProposalCoverage,
    covering_books_for_chart: list[BookCoverage],
    local_projection: LocalProjection2D,
) -> str:
    out: list[str] = []
    out.append("\\clearpage")
    out.append("\\begingroup\\small")
    topic_url = f"https://openalex.org/topics/{topic_id}"
    topic_link = (
        f"\\href{{{topic_url}}}{{\\texttt{{{latex_escape(topic_id)}}}}}"
    )
    out.append(
        f"\\subsection[{tex(topic_name)} ({latex_escape(topic_id)})]"
        f"{{{tex(topic_name)} ({topic_link})}}"
    )
    out.append(f"\\label{{appendix:proposal-{latex_escape(topic_id)}}}")
    pct_str = (
        f"{gap_size_pct:.2f}" if gap_size_pct is not None else "n/a"
    )
    out.append(
        f"\\noindent\\textit{{Topic has {num_books} books and "
        f"{num_articles} articles; proposal covers "
        f"{gap_coverage * 100:.1f}\\% of the target gap; "
        f"target-gap size percentile = {pct_str}.}}"
    )
    out.append("\\par\\medskip")

    # Two figures, centered.
    out.append("\\begin{center}")
    out.append(_render_bar_chart_tikz(
        book_labels=[tex(b.display_name, 40) for b in covering_books_for_chart],
        pre_counts=[b.articles_covered for b in covering_books_for_chart],
        post_counts=[
            b.articles_covered_after_proposal for b in covering_books_for_chart
        ],
        proposal_post_count=proposal_coverage.articles_covered,
    ))
    out.append("\\quad")
    out.append(_render_projection_scatter_tikz(
        local_projection,
        book_coverages=[b.articles_covered for b in covering_books_for_chart],
    ))
    out.append("\\end{center}")
    out.append("\\par\\medskip")

    out.append(f"\\paragraph{{{_gray('Book proposal')}}}")
    out.append(f"\\textbf{{{tex(prop_title)}}}")
    out.append("")
    out.append(tex(prop_abstract))

    out.append(
        f"\\paragraph{{{_gray('Prompt articles')}}}"
    )
    out.append("\\begin{enumerate}\\setlength\\itemsep{0.25em}")
    for a in articles:
        link = _openalex_href(a.openalex_id)
        url_part = f" ({link})" if link else ""
        out.append(
            f"\\item \\textbf{{\\textsf{{{tex(a.display_name, _PROMPT_TITLE_MAX)}}}}}{url_part}"
        )
        out.append("")
        if _PROMPT_ABSTRACT_MAX > 0:
            out.append(tex(a.abstract, _PROMPT_ABSTRACT_MAX))
    out.append("\\end{enumerate}")

    out.append(f"\\paragraph{{{_gray('Highest-coverage existing book')}}}")
    if top_book is None:
        out.append("\\textit{No books found in this topic.}")
    else:
        link = _openalex_href(top_book.openalex_id)
        url_part = f" ({link})" if link else ""
        out.append(f"\\textbf{{{tex(top_book.display_name)}}}{url_part}")
        out.append("")
        pre = top_book.articles_covered
        post = top_book.articles_covered_after_proposal
        pre_pct = top_book.fraction_covered * 100
        # post_pct = post / gap_size * 100. Since fraction = articles/gap_size,
        # post_pct = pre_pct * (post / pre) when pre > 0.
        post_pct = pre_pct * (post / pre) if pre > 0 else 0.0
        out.append(
            f"\\noindent\\textit{{Covers {pre} {{\\color{{gray}}\\textrightarrow}} {post} articles, "
            f"{pre_pct:.2f}\\% {{\\color{{gray}}\\textrightarrow}} {post_pct:.2f}\\% of the gap "
            f"after the proposal is added as a book.}}"
        )
        out.append("\\par\\medskip")
        out.append(tex(top_book.abstract, _BOOK_ABSTRACT_MAX))

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
        top_book = context.covering_books[0] if context.covering_books else None

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
                top_book=top_book,
                proposal_coverage=context.proposal_coverage,
                covering_books_for_chart=context.covering_books,
                local_projection=context.local_projection_2d,
            )
        )

    preamble = (
        "% Auto-generated by artifacts_scripts/generate_proposal_appendix.py "
        "-- do not edit by hand.\n"
        "\\providecommand{\\unkchar}{\\rule[0.1ex]{0.55em}{0.55em}}\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(preamble + "\n" + "\n\n".join(chunks) + "\n")
    print(f"Wrote {len(chunks)} topic pages to {args.output}")


if __name__ == "__main__":
    main()
