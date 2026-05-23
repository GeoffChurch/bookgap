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


_PROMPT_TITLE_MAX = 80
_PROMPT_ABSTRACT_MAX = 200
_BOOK_ABSTRACT_MAX = 600  # highest-coverage book abstract (one per topic)


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
) -> str:
    out: list[str] = []
    out.append("\\clearpage")
    out.append("\\begingroup\\small")
    out.append(
        f"\\subsection{{{tex(topic_name)} "
        f"(\\texttt{{{latex_escape(topic_id)}}})}}"
    )
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

    out.append("\\paragraph{Proposed textbook.}")
    out.append(f"\\textbf{{{tex(prop_title)}}}")
    out.append("")
    out.append(tex(prop_abstract))

    out.append("\\paragraph{Prompt articles (the ten randomly sampled from the gap).}")
    out.append("\\begin{enumerate}\\setlength\\itemsep{0.25em}")
    for a in articles:
        url_part = f" (\\url{{{a.openalex_id}}})" if a.openalex_id else ""
        out.append(
            f"\\item \\textbf{{{tex(a.display_name, _PROMPT_TITLE_MAX)}}}{url_part}"
        )
        out.append("")
        out.append(tex(a.abstract, _PROMPT_ABSTRACT_MAX))
    out.append("\\end{enumerate}")

    out.append("\\paragraph{Highest-coverage existing book.}")
    if top_book is None:
        out.append("\\textit{No books found in this topic.}")
    else:
        url_part = f"\\url{{{top_book.openalex_id}}}; " if top_book.openalex_id else ""
        out.append(
            f"\\textbf{{{tex(top_book.display_name)}}} "
            f"({url_part}covers {top_book.articles_covered} articles, "
            f"{top_book.fraction_covered * 100:.2f}\\% of the gap)"
        )
        out.append("")
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
