#!/usr/bin/env python
# artifacts_scripts/curriculum/eval_book_judge.py
"""Single-blind gemma4 LLM-judge for a generated curriculum, scored as a *book*.

This is a SECOND extrinsic LLM-as-judge layer, complementary to
``curriculum_llm_judge.py`` (which audits per-topic *validity / keyword-noise /
ordering* on the project's QC-cache curricula). Here we instead render a single
built curriculum as a SINGLE-BLIND "book outline" — a title plus its ordered
chapters (the curriculum's sub-topics in learning-path order, each with its top
keywords and a few representative work titles) — and ask gemma4:31b to score it
on a 4-dimension 1-7 rubric, PER CHAPTER and for the BOOK overall.

Single-blind means the rendered outline carries NO config / best / baseline /
which-run label anywhere: the judge sees only the book title and chapters, never
the directory name, the selection variant, or any "this is the good one" hint.
The parent orchestrator runs this once per curriculum and compares the resulting
JSON across runs; the judge must not be able to tell the runs apart by anything
other than the curriculum content itself.

Reused infrastructure
=====================
The ollama plumbing is imported wholesale from ``curriculum_llm_judge``:
``JudgeClient`` (model-keyed ``ollama.chat`` with ``format=<schema>`` structured
output, tolerant balanced-``{...}`` extraction, persist-raw-on-parse-failure to a
model-keyed ``*_rawfails.jsonl``, and call/failure counters) and
``_extract_first_json_object`` / ``_log``. We add only the book-outline rendering,
the 4-dim rubric schema/prompts, and the per-(chapter,book)x(repeat) scoring loop
with a per-dimension median over the repeat-votes.

The 4-dimension rubric (1-7 Likert, each with a one-sentence "why")
==================================================================
  - **interestingness**   — does it seem an interesting book worth reading
  - **comprehensiveness** — does it seem to cover the full subject
  - **specificity**       — does it maintain focus on the proposed topic
  - **progressive**       — does the material move from easy to hard

Each item (every chapter, and the book overall) is judged ``--repeats`` times
(default 3); we report every per-vote score and the per-dimension MEDIAN.

Input: a built curriculum directory under ``artifacts/curriculum/<run>/`` holding
``summary.md`` (the rich source: per-topic "Top Keywords" + "Top Works" sections
in curriculum/rank order) and ``pipeline.log`` (carries the human-readable topic
name, used as the blind book title).

Usage
=====
    uv run python -m artifacts_scripts.curriculum.eval_book_judge \
        --run-dir artifacts/curriculum/gap_curriculum_T10181_base

    # custom model / repeats / output path:
    uv run python -m artifacts_scripts.curriculum.eval_book_judge \
        --run-dir artifacts/curriculum/gap_curriculum_T11902 \
        --model gemma4:31b --repeats 3 \
        --out artifacts/curriculum_eval/gap_curriculum_T11902.json

Output: one JSON per curriculum under ``artifacts/curriculum_eval/<run>.json``:
    {topic, run_dir, model, chapters:[{idx,label,dims:{dim:{scores,median,comments}}}],
     book:{dims:{...}}}
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ollama

# Reuse the existing ollama judge plumbing wholesale (model-keyed structured
# call, tolerant parse, persist-raw-on-failure, call/fail counters).
from artifacts_scripts.curriculum.curriculum_llm_judge import (  # noqa: F401
    JudgeClient,
    _extract_first_json_object,
    _log,
)

DEFAULT_MODEL = "gemma4:31b"
EVAL_OUT_DIR = Path("artifacts/curriculum_eval")

# The four scored dimensions, with the rubric anchor text shown to the judge.
DIMENSIONS: List[Tuple[str, str]] = [
    ("interestingness",
     "Interestingness: does this seem an interesting book worth reading? "
     "(7 = compelling, you would want to read it; 1 = dull, not worth reading.)"),
    ("comprehensiveness",
     "Comprehensiveness: does it seem to cover the full subject? "
     "(7 = covers the subject thoroughly with few obvious gaps; "
     "1 = large important parts of the subject are missing.)"),
    ("specificity",
     "Specificity: does it maintain focus on the proposed topic? "
     "(7 = every chapter stays tightly on the stated topic; "
     "1 = wanders into unrelated material, off-topic.)"),
    ("progressive",
     "Progressive: does the material move from easy to hard / foundations to "
     "advanced? (7 = a clear easy-to-hard learning progression; "
     "1 = ordering is jumbled, advanced material before its prerequisites.)"),
]
_DIM_KEYS = [d[0] for d in DIMENSIONS]


# ---------------------------------------------------------------------------
# Parsing a built curriculum directory into an ordered list of chapters
# ---------------------------------------------------------------------------

@dataclass
class Chapter:
    """One curriculum sub-topic, in learning-path order, as a book chapter."""
    idx: int                 # 1-based chapter number (= curriculum rank)
    label: str               # short sub-topic label (from its top keywords)
    keywords: List[str]
    titles: List[str]        # representative work titles


@dataclass
class BookOutline:
    topic: str               # blind book title (the topic name)
    chapters: List[Chapter]


def _parse_topic_name(run_dir: Path) -> str:
    """Recover the human-readable topic name (the blind book title).

    ``pipeline.log`` logs e.g. ``Topic T10181 (Natural Language Processing
    Techniques) gap 0: ...`` — the parenthesised name is config-agnostic and
    makes the best single-blind title. Falls back to the bare topic id parsed
    from the directory name (``gap_curriculum_T10181_base`` -> ``T10181``), and
    finally to the directory name itself.
    """
    log = run_dir / "pipeline.log"
    if log.exists():
        for ln in log.read_text(errors="ignore").splitlines():
            m = re.search(r"Topic\s+(T\d+)\s+\(([^)]+)\)", ln)
            if m:
                return m.group(2).strip()
            if "Topic" in ln and "(" in ln:  # cheap early-out scan limiter
                pass
    m = re.search(r"(T\d+)", run_dir.name)
    return m.group(1) if m else run_dir.name


def _split_md_row(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _label_from_keywords(keywords: List[str]) -> str:
    """A short human chapter label from the top keywords (underscores -> spaces)."""
    top = [k.replace("_", " ") for k in keywords[:3]]
    return ", ".join(top) if top else "(unlabelled)"


def _parse_chapters_from_summary(summary_path: Path) -> List[Chapter]:
    """Parse per-topic Top Keywords + Top Works from a curriculum ``summary.md``.

    The file emits, in curriculum (rank) order, one ``## Topic N`` section each
    with a ``### Top Keywords`` markdown table (``| Keyword | Score |``) and a
    ``### Top Works`` section whose entries are ``#### k. <full title>`` headers.
    We read the FULL untruncated work titles from those ``####`` headers (the
    Overview table truncates them), and take keywords from the per-topic table.
    """
    lines = summary_path.read_text().splitlines()
    # Index the start line of each "## Topic N" section.
    sec_starts: List[Tuple[int, int]] = []  # (line_idx, topic_number)
    for i, ln in enumerate(lines):
        m = re.match(r"^##\s+Topic\s+(\d+)\s*$", ln.strip())
        if m:
            sec_starts.append((i, int(m.group(1))))
    if not sec_starts:
        raise ValueError(f"no '## Topic N' sections found in {summary_path}")

    chapters: List[Chapter] = []
    for si, (start, topic_no) in enumerate(sec_starts):
        end = sec_starts[si + 1][0] if si + 1 < len(sec_starts) else len(lines)
        block = lines[start:end]
        keywords = _parse_top_keywords_block(block)
        titles = _parse_top_works_block(block)
        chapters.append(Chapter(
            idx=topic_no,
            label=_label_from_keywords(keywords),
            keywords=keywords,
            titles=titles,
        ))
    chapters.sort(key=lambda c: c.idx)
    # Renumber 1..N so chapter index is contiguous learning-path position.
    for new_idx, ch in enumerate(chapters, start=1):
        ch.idx = new_idx
    return chapters


def _parse_top_keywords_block(block: List[str]) -> List[str]:
    """Pull keyword cells from the ``### Top Keywords`` markdown table in a block."""
    kws: List[str] = []
    in_kw = False
    for ln in block:
        s = ln.strip()
        if s.startswith("### Top Keywords"):
            in_kw = True
            continue
        if in_kw:
            if s.startswith("###") or s.startswith("## "):
                break
            if not s.startswith("|"):
                continue
            cells = _split_md_row(s)
            if not cells:
                continue
            head = cells[0].lower()
            if head in ("keyword", "") or set(cells[0]) <= set(":- "):
                continue  # header / separator row
            kws.append(cells[0])
    return kws


def _parse_top_works_block(block: List[str]) -> List[str]:
    """Pull full work titles from ``#### k. <title>`` headers in a block."""
    titles: List[str] = []
    in_works = False
    for ln in block:
        s = ln.strip()
        if s.startswith("### Top Works"):
            in_works = True
            continue
        if in_works:
            if s.startswith("### ") or s.startswith("## "):
                break
            m = re.match(r"^####\s+\d+\.\s+(.*)$", s)
            if m:
                title = m.group(1).strip()
                if title:
                    titles.append(title)
    return titles


def load_book_outline(run_dir: Path) -> BookOutline:
    summary = run_dir / "summary.md"
    if not summary.exists():
        raise FileNotFoundError(f"no summary.md in {run_dir}")
    topic = _parse_topic_name(run_dir)
    chapters = _parse_chapters_from_summary(summary)
    if not chapters:
        raise ValueError(f"parsed zero chapters from {summary}")
    return BookOutline(topic=topic, chapters=chapters)


# ---------------------------------------------------------------------------
# Single-blind outline rendering
# ---------------------------------------------------------------------------

def render_outline(run_dir: Path, n_titles: int = 5, n_keywords: int = 30) -> str:
    """Render a curriculum dir as a SINGLE-BLIND book outline string.

    Title + each chapter in order (index, sub-topic label, top keywords, and 3-5
    representative work titles). Carries NO config/best/baseline/run label.
    """
    book = load_book_outline(Path(run_dir))
    out: List[str] = [f"Book title: {book.topic}", ""]
    out.append("Proposed chapters (in order):")
    for ch in book.chapters:
        kw = ", ".join(ch.keywords[:n_keywords]) or "(none)"
        titles = ch.titles[:n_titles]
        out.append("")
        out.append(f"Chapter {ch.idx}: {ch.label}")
        out.append(f"  Key topics/keywords: {kw}")
        if titles:
            out.append("  Representative works:")
            for t in titles:
                out.append(f"    - {t}")
        else:
            out.append("  Representative works: (none listed)")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Rubric scoring (structured JSON, repeat-voted, median-aggregated)
# ---------------------------------------------------------------------------

# Force structured JSON: {dim: {score:int 1-7, comment:str}} for all 4 dims.
def _rubric_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            dim: {
                "type": "object",
                "properties": {
                    "score": {"type": "integer", "minimum": 1, "maximum": 7},
                    "comment": {"type": "string"},
                },
                "required": ["score", "comment"],
            }
            for dim in _DIM_KEYS
        },
        "required": _DIM_KEYS,
    }


_RUBRIC_TEXT = "\n".join(f"  - {anchor}" for _, anchor in DIMENSIONS)

_BOOK_SYSTEM = (
    "You are an expert book reviewer and curriculum designer evaluating the "
    "proposed outline of a new textbook. Judge the outline on its own merits and "
    "respond ONLY with JSON matching the requested schema."
)


def _book_user(outline: str) -> str:
    return (
        "Here is the proposed outline of a textbook (its title, then its chapters "
        "in order, each with key topics and a few representative works):\n\n"
        f"{outline}\n\n"
        "Considering the WHOLE BOOK, rate it on each of the following four "
        "dimensions, on a scale of 1 (worst) to 7 (best):\n"
        f"{_RUBRIC_TEXT}\n\n"
        "For each dimension give an integer score from 1 to 7 and a one-sentence "
        "reason ('comment')."
    )


def _chapter_user(book_title: str, ch: Chapter, n_titles: int = 5,
                  n_keywords: int = 30) -> str:
    kw = ", ".join(ch.keywords[:n_keywords]) or "(none)"
    titles = "; ".join(ch.titles[:n_titles]) or "(none)"
    return (
        f"This is one chapter of a textbook titled \"{book_title}\".\n\n"
        f"Chapter {ch.idx}: {ch.label}\n"
        f"  Key topics/keywords: {kw}\n"
        f"  Representative works: {titles}\n\n"
        "Considering THIS CHAPTER (in the context of such a book), rate it on "
        "each of the following four dimensions, on a scale of 1 (worst) to 7 "
        f"(best):\n{_RUBRIC_TEXT}\n\n"
        "For each dimension give an integer score from 1 to 7 and a one-sentence "
        "reason ('comment')."
    )


def _parse_rubric(obj: Optional[dict]) -> Optional[Dict[str, Tuple[int, str]]]:
    """Validate a rubric response -> {dim: (score, comment)} or None."""
    if not isinstance(obj, dict):
        return None
    out: Dict[str, Tuple[int, str]] = {}
    for dim in _DIM_KEYS:
        cell = obj.get(dim)
        if not isinstance(cell, dict) or "score" not in cell:
            return None
        try:
            score = int(cell["score"])
        except (ValueError, TypeError):
            return None
        score = max(1, min(7, score))
        out[dim] = (score, str(cell.get("comment", "")))
    return out


@dataclass
class ItemScore:
    """Repeat-voted rubric scores for one item (a chapter or the whole book)."""
    # dim -> list of per-vote scores; dim -> list of per-vote comments
    scores: Dict[str, List[int]] = field(default_factory=lambda: {d: [] for d in _DIM_KEYS})
    comments: Dict[str, List[str]] = field(default_factory=lambda: {d: [] for d in _DIM_KEYS})

    def add(self, rubric: Dict[str, Tuple[int, str]]) -> None:
        for dim, (sc, cm) in rubric.items():
            self.scores[dim].append(sc)
            self.comments[dim].append(cm)

    def to_dims_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for dim in _DIM_KEYS:
            scs = self.scores[dim]
            out[dim] = {
                "scores": scs,
                "median": (float(statistics.median(scs)) if scs else None),
                "comments": self.comments[dim],
            }
        return out


def _judge_item(client: JudgeClient, system: str, user: str, repeats: int,
                ctx: Dict[str, Any]) -> ItemScore:
    item = ItemScore()
    schema = _rubric_schema()
    for rep in range(repeats):
        client.set_context(**ctx, repeat=rep)
        obj = client.ask(system, user, schema, dimension=ctx.get("kind", "item"))
        rubric = _parse_rubric(obj)
        if rubric is None:
            # ask() already counted+dumped a transport/parse failure if obj is
            # None; a schema-shape miss on a parsed dict is counted here.
            if obj is not None:
                client.n_parse_fail += 1
            continue
        item.add(rubric)
    return item


def evaluate_book(run_dir: Path, client: JudgeClient, *, repeats: int,
                  n_titles: int = 5, n_keywords: int = 30) -> Dict[str, Any]:
    book = load_book_outline(run_dir)
    outline = render_outline(run_dir, n_titles=n_titles, n_keywords=n_keywords)
    _log(f"judging book '{book.topic}': {len(book.chapters)} chapters + 1 book "
         f"overall, x{repeats} each ({client.model})")

    chapter_results: List[Dict[str, Any]] = []
    for ch in book.chapters:
        item = _judge_item(
            client, _BOOK_SYSTEM,
            _chapter_user(book.topic, ch, n_titles, n_keywords),
            repeats, {"kind": "chapter", "chapter_idx": ch.idx, "topic": book.topic},
        )
        chapter_results.append({
            "idx": ch.idx,
            "label": ch.label,
            "keywords": ch.keywords[:n_keywords],
            "titles": ch.titles[:n_titles],
            "dims": item.to_dims_dict(),
        })

    book_item = _judge_item(
        client, _BOOK_SYSTEM, _book_user(outline), repeats,
        {"kind": "book", "topic": book.topic},
    )

    return {
        "topic": book.topic,
        "run_dir": str(run_dir),
        "model": client.model,
        "repeats": repeats,
        "n_chapters": len(book.chapters),
        "chapters": chapter_results,
        "book": {"dims": book_item.to_dims_dict()},
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(result: Dict[str, Any]) -> None:
    print()
    print(f"Book: {result['topic']}  ({result['n_chapters']} chapters, "
          f"model={result['model']}, repeats={result['repeats']})")
    hdr = f"{'item':<28}" + "".join(f"{d[:6]:>9}" for d in _DIM_KEYS)
    print(hdr)
    print("-" * len(hdr))
    for ch in result["chapters"]:
        row = f"{('Ch%d ' % ch['idx'] + ch['label'])[:28]:<28}"
        for dim in _DIM_KEYS:
            med = ch["dims"][dim]["median"]
            row += f"{('n/a' if med is None else f'{med:.1f}'):>9}"
        print(row)
    row = f"{'BOOK (overall)':<28}"
    for dim in _DIM_KEYS:
        med = result["book"]["dims"][dim]["median"]
        row += f"{('n/a' if med is None else f'{med:.1f}'):>9}"
    print(row)
    print("\n(scores are per-dimension MEDIANS over repeat-votes, 1-7; "
          "full per-vote scores + comments are in the JSON.)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="a built curriculum directory under "
                         "artifacts/curriculum/<run>/ (must hold summary.md).")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"ollama model to PIN for all judgments "
                         f"(default {DEFAULT_MODEL}).")
    ap.add_argument("--repeats", type=int, default=3,
                    help="repeat-vote each item (each chapter + the book) this "
                         "many times; the per-dimension median is reported "
                         "(default 3).")
    ap.add_argument("--n-titles", type=int, default=5,
                    help="representative work titles to show per chapter (3-5; "
                         "default 5).")
    ap.add_argument("--n-keywords", type=int, default=30,
                    help="keywords to show per chapter (default 30).")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-predict", type=int, default=1536,
                    help="max tokens per judge call (default 1536; 512 truncates "
                         "gemma4:31b's verbose structured output).")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON path (default "
                         "artifacts/curriculum_eval/<run>.json).")
    ap.add_argument("--print-outline", action="store_true",
                    help="print the rendered single-blind outline and exit "
                         "(no judging).")
    args = ap.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        _log(f"FATAL: run dir {run_dir} does not exist")
        sys.exit(2)

    if args.print_outline:
        print(render_outline(run_dir, n_titles=args.n_titles,
                             n_keywords=args.n_keywords))
        return

    repeats = max(1, args.repeats)

    # Verify ollama is reachable + the model is pulled before spinning.
    try:
        local = {m.model for m in ollama.list().models if m.model}
    except Exception as e:
        _log(f"FATAL: cannot reach ollama server ({e}). Is `ollama serve` up?")
        sys.exit(2)
    if args.model not in local:
        _log(f"FATAL: model '{args.model}' not pulled. Available: "
             f"{sorted(local)}. Run `ollama pull {args.model}`.")
        sys.exit(2)

    out_path = args.out or (EVAL_OUT_DIR / f"{run_dir.name}.json")
    model_slug = re.sub(r"[^A-Za-z0-9]+", "_", args.model).strip("_")
    rawfail_path = EVAL_OUT_DIR / f"{run_dir.name}_{model_slug}_rawfails.jsonl"
    if rawfail_path.exists():
        rawfail_path.unlink()

    client = JudgeClient(model=args.model, temperature=args.temperature,
                         num_predict=args.num_predict, rawfail_path=rawfail_path)

    t0 = time.time()
    result = evaluate_book(run_dir, client, repeats=repeats,
                           n_titles=args.n_titles, n_keywords=args.n_keywords)
    elapsed = time.time() - t0
    fail_rate = client.n_parse_fail / max(1, client.n_calls)
    _log(f"done: {client.n_calls} ollama calls, {client.n_parse_fail} "
         f"parse/transport failures ({fail_rate:.1%}), {elapsed:.0f}s "
         f"({elapsed / max(1, client.n_calls):.1f}s/call)")
    if client.n_parse_fail and rawfail_path.exists():
        _log(f"raw failures dumped to {rawfail_path}")

    result["n_ollama_calls"] = client.n_calls
    result["n_parse_failures"] = client.n_parse_fail
    result["parse_failure_rate"] = fail_rate
    result["elapsed_seconds"] = elapsed

    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    out_path.write_text(json.dumps(result, indent=2))
    _log(f"wrote {out_path}")

    _print_summary(result)


if __name__ == "__main__":
    main()
