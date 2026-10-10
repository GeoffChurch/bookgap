#!/usr/bin/env python3
"""
Validate that the survey question text matches the titles and abstracts in
selected_books.csv, and emit a LaTeX table summarizing the book choices per
question type.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from artifacts_scripts.common import latex_provenance

QuestionType = str

QUESTION_SLUGS = {
    "interesting": "interesting?",
    "well_written": "well-written?",
    "publishable": "worth publishing?",
}

EXPECTED_SUFFIX = "\n\n<br><br><b>Do you personally think this book is...</b>"

# Subcolumn labels used across the three question groups: the more-cited book, the
# less-cited book, and our generated proposal.
#
# Words, not arrows. These were \texttt{↑}/\texttt{↓}, whose literal characters inside
# \texttt resolve to TS1 glyphs shipping only as 600dpi PK bitmaps -- pdflatex then
# embeds a Type 3 font, the glyphs print fuzzy, and pdffonts reports `uni no`, meaning
# they survive neither copy-paste nor a screen reader (ACL asks authors to "prioritise
# the accessibility of your paper"). Words dodge that entirely and self-describe.
#
# Width cost, since it is not obvious: these are 4 monospace characters (26.2pt at
# 11pt) against 19.1pt of data beneath, so in the per-topic table they take over as the
# thing setting each column's width, adding ~7.1pt to each of the six non-`ours`
# columns. That table is consequently set \small; see the comment at its float in
# main.tex. The compact summary table is unaffected -- `ours` already set that width.
SUBCOLUMN_LABELS = (r"\texttt{more}", r"\texttt{less}", r"\texttt{ours}")

# Cell order shared by both tables: criterion-major, then book class.
COLUMN_ORDER: Tuple[Tuple[str, str], ...] = (
    ("interesting", "high"),
    ("interesting", "low"),
    ("interesting", "generated"),
    ("well_written", "high"),
    ("well_written", "low"),
    ("well_written", "generated"),
    ("publishable", "high"),
    ("publishable", "low"),
    ("publishable", "generated"),
)

CRITERION_HEADERS = ("Interesting?", "Well-written?", "Publishable?")

# Two-sided 95% t critical values, indexed by degrees of freedom.
T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
}

QUESTION_PATTERN = re.compile(
    r"Title:\s*(?P<title>.*?)\s*Abstract:\s*(?P<abstract>.*?)\s*Do you personally think this book is\.\.\.\s*-\s*(?P<question>interesting\?|well-written\?|worth publishing\?)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class BookRecord:
    topic_id: str
    topic_name: str
    book_id: Optional[str]
    kernel_percentile: Optional[float]
    title: str
    abstract: str


@dataclass
class SurveyQuestion:
    column_name: str
    question_type: QuestionType
    title: str
    abstract: str


def parse_question_cell(text: str) -> Optional[Tuple[str, str, QuestionType]]:
    """Parse a survey question cell into (title, abstract, question_type)."""
    if not text or "Title:" not in text:
        return None
    match = QUESTION_PATTERN.search(text)
    if not match:
        return None
    title = match.group("title").strip()
    abstract = match.group("abstract").strip()
    question_raw = match.group("question").strip().lower()
    for slug, needle in QUESTION_SLUGS.items():
        if question_raw == needle:
            return title, abstract, slug
    return None


def strip_html(text: str) -> str:
    """Remove HTML tags and normalize non-breaking spaces."""
    clean = re.sub("<.*?>", "", text or "")
    return clean.replace("&nbsp;", " ").strip()


def normalize_text(text: str) -> str:
        """Normalize abstract/title text for strict comparison.

        - Convert literal "\n" sequences into real newlines so CSV-escaped text
            matches rendered survey text.
        - Strip trailing whitespace for stability.
        - Replace common HTML non-breaking space entity with a plain space.
        """
        return text.replace("&nbsp;", " ").replace("\\n", "\n").rstrip()


def load_selected_books(path: Path) -> List[BookRecord]:
    rows: List[BookRecord] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel_str = row.get("kernel_percentile") or ""
            kernel = float(kernel_str) if kernel_str else None
            rows.append(
                BookRecord(
                    topic_id=row["topic_id"],
                    topic_name=row["topic_name"],
                    book_id=row.get("book_id") or None,
                    kernel_percentile=kernel,
                    title=normalize_text(row["title"].strip()),
                    abstract=normalize_text(row["abstract"].strip()),
                )
            )
    return rows


def load_qsf_questions(path: Path) -> Dict[str, Tuple[str, str, List[QuestionType]]]:
    """Parse Qualtrics QSF to map question_id -> (title, abstract, question_types).

    Some QSF entries include the matrix choices (interesting / well-written /
    worth publishing) inline rather than as question text; in those cases we
    treat it as having all three question types.
    """

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, Tuple[str, str, List[QuestionType]]] = {}
    for element in data.get("SurveyElements", []):
        if element.get("Element") != "SQ":
            continue
        payload = element.get("Payload", {})
        q_id = payload.get("QuestionID")
        q_text = payload.get("QuestionText", "")
        if not q_id or "Abstract" not in q_text or "Title" not in q_text:
            continue

        # Keep the question tail so we can recover the question type; just strip HTML.
        clean_text = strip_html(q_text)
        parsed = parse_question_cell(clean_text)
        if parsed:
            title, abstract, qtype = parsed
            out[str(q_id)] = (normalize_text(title), normalize_text(abstract), [qtype])
            continue

        # Fallback: parse Title/Abstract without the trailing question text.
        if "Title:" in clean_text and "Abstract:" in clean_text:
            # Split once on Abstract to avoid greedy capture.
            try:
                title_part, rest = clean_text.split("Abstract:", 1)
                title = title_part.replace("Title:", "", 1).strip()
                abstract = rest.strip()
                out[str(q_id)] = (normalize_text(title), normalize_text(abstract), list(QUESTION_SLUGS.keys()))
            except ValueError:
                continue
    return out


def load_survey(path: Path) -> Tuple[List[SurveyQuestion], List[Dict[str, str]]]:
    """Load survey questions and response rows.

    Returns the parsed questions plus a list of response dicts keyed by column name.
    The CSV is expected to have the first row as headers and the second row as the
    human-readable question text; subsequent rows are responses.
    """
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            question_row = next(reader)
        except StopIteration as exc:  # pragma: no cover - defensive
            raise ValueError("Survey CSV must have at least two rows") from exc

        questions: List[SurveyQuestion] = []
        for col_name, cell in zip(header, question_row):
            parsed = parse_question_cell(cell)
            if not parsed:
                continue
            title, abstract, qtype = parsed
            questions.append(
                SurveyQuestion(
                    column_name=col_name,
                    question_type=qtype,
                    title=normalize_text(title),
                    abstract=normalize_text(abstract),
                )
            )

        responses: List[Dict[str, str]] = []
        for row in reader:
            if not any(cell.strip() for cell in row):
                continue
            responses.append({col: val for col, val in zip(header, row)})

    return questions, responses


def validate_alignment(selected: List[BookRecord], survey_questions: List[SurveyQuestion]) -> None:
    """Ensure bidirectional exact matches between selected_books and survey text."""
    survey_by_title: Dict[str, Dict[QuestionType, SurveyQuestion]] = defaultdict(dict)
    for q in survey_questions:
        survey_by_title[q.title][q.question_type] = q

    selected_by_title: Dict[str, BookRecord] = {b.title: b for b in selected}

    missing_messages = []
    # Every selected book must appear with all three question types, and abstract must match exactly.
    for book in selected:
        questions = survey_by_title.get(book.title)
        if not questions:
            missing_messages.append(f"Missing in survey: {book.title}")
            continue
        for qtype in QUESTION_SLUGS:
            if qtype not in questions:
                missing_messages.append(f"Missing question '{qtype}' for {book.title}")
        # Check abstract equality for one representative question.
        sample_question = next(iter(questions.values()))
        if book.abstract != sample_question.abstract:
            prefix = 0
            for a, b in zip(book.abstract, sample_question.abstract):
                if a != b:
                    break
                prefix += 1
            snippet_sel = book.abstract[prefix:prefix + 80].replace("\n", "\\n")
            snippet_sur = sample_question.abstract[prefix:prefix + 80].replace("\n", "\\n")
            missing_messages.append(
                f"Abstract mismatch for {book.title}\n"
                f"\tselected length={len(book.abstract)}\n"
                f"\tsurvey length={len(sample_question.abstract)}\n"
                f"\tfirst difference at position {prefix}\n"
                f"\tselected snippet: {snippet_sel}\n"
                f"\tsurvey snippet:   {snippet_sur}"
            )
    if missing_messages:
        raise ValueError("\n".join(missing_messages))

    # Every survey question must correspond to a selected book title.
    extras = []
    for title in survey_by_title:
        if title not in selected_by_title:
            extras.append(f"Survey includes unknown title: {title}")
    if extras:
        raise ValueError("\n".join(extras))


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
        # Unicode characters that appear in OpenAlex titles/abstracts but aren't
        # in T1 fontenc; map them to LaTeX equivalents to keep the appendix
        # compilable with pdflatex.
        " ": "~",
        "‑": "-",
        "–": "--",
        "—": "---",
        "‘": "`",
        "’": "'",
        "“": "``",
        "”": "''",
        "α": r"$\alpha$",
        "β": r"$\beta$",
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def choose_by_percentile(records: List[BookRecord]) -> Tuple[Optional[BookRecord], Optional[BookRecord], Optional[BookRecord]]:
    numeric = [r for r in records if r.kernel_percentile is not None]
    generated = next((r for r in records if r.kernel_percentile is None or r.book_id is None), None)
    if numeric:
        # Safe because we filtered out None kernel_percentile values.
        high = max(numeric, key=lambda r: r.kernel_percentile or float("-inf"))
        low = min(numeric, key=lambda r: r.kernel_percentile or float("inf"))
    else:
        high = None
        low = None
    return high, low, generated


def compute_question_means(
    questions: List[SurveyQuestion], responses: List[Dict[str, str]]
) -> Dict[Tuple[str, QuestionType], Optional[float]]:
    """Average numeric survey responses per (title, question_type)."""
    means: Dict[Tuple[str, QuestionType], Optional[float]] = {}
    for q in questions:
        vals: List[float] = []
        for row in responses:
            raw = row.get(q.column_name, "").strip()
            if not raw:
                continue
            try:
                vals.append(float(raw))
            except ValueError:
                continue
        means[(q.title, q.question_type)] = sum(vals) / len(vals) if vals else None
    return means


def load_llm_results(
    llm_path: Path,
    qsf_path: Path,
    models: Optional[List[str]] = None,
    prompt_modes: Optional[List[str]] = None,
) -> Dict[Tuple[str, str], Dict[Tuple[str, QuestionType], float]]:
    """Load LLM scores and return per-variant means keyed by (title, question_type).

    Variants are keyed by (model, prompt_mode). Filters:
    - models: only include these model specs if provided.
    - prompt_modes: only include these prompt modes if provided.
    """

    with llm_path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("LLM results must be a list of records")

    qsf_map = load_qsf_questions(qsf_path)
    score_key_map = {
        "interesting": "interesting",
        "well_written": "well_written",
        "worth_publishing": "publishable",
    }

    bucket: Dict[Tuple[str, str], Dict[Tuple[str, QuestionType], List[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        qid = str(row.get("question_id"))
        model = row.get("model")
        eval_block = row.get("evaluation")
        prompt_mode = row.get("prompt_mode")
        if not prompt_mode:
            prompt_mode = "standard"

        if not qid or not model or not isinstance(eval_block, dict):
            continue
        if models and model not in models:
            continue
        if prompt_modes and prompt_mode not in prompt_modes:
            continue

        mapped = qsf_map.get(qid)
        if not mapped:
            continue
        title, _abstract, qtypes = mapped

        for raw_key, norm_key in score_key_map.items():
            if norm_key not in qtypes:
                continue
            val = eval_block.get(raw_key)
            if isinstance(val, (int, float)):
                bucket[(model, prompt_mode)][(title, norm_key)].append(float(val))

    means: Dict[Tuple[str, str], Dict[Tuple[str, QuestionType], float]] = {}
    for variant, scores in bucket.items():
        means[variant] = {
            key: sum(vals) / len(vals)
            for key, vals in scores.items()
            if vals
        }
    return means


def collect_cell_values(
    selected: List[BookRecord],
    means: Dict[Tuple[str, QuestionType], Optional[float]],
) -> Tuple[List[Tuple[str, str, List[Optional[float]]]], List[List[float]]]:
    """Per-topic mean ratings in COLUMN_ORDER, plus the per-column value lists.

    Both the compact main-text table and the per-topic appendix table are
    rendered from this single pass, so their numbers cannot drift apart.
    """
    by_topic: Dict[str, List[BookRecord]] = defaultdict(list)
    topic_names: Dict[str, str] = {}
    for rec in selected:
        by_topic[rec.topic_id].append(rec)
        topic_names[rec.topic_id] = rec.topic_name

    rows: List[Tuple[str, str, List[Optional[float]]]] = []
    column_sums: List[List[float]] = [[] for _ in COLUMN_ORDER]

    for topic_id in sorted(by_topic.keys()):
        high, low, generated = choose_by_percentile(by_topic[topic_id])
        by_class = {"high": high, "low": low, "generated": generated}

        row_values: List[Optional[float]] = []
        for qtype, cls in COLUMN_ORDER:
            chosen = by_class[cls]
            row_values.append(means.get((chosen.title, qtype)) if chosen else None)

        for idx, val in enumerate(row_values):
            if val is not None:
                column_sums[idx].append(val)
        rows.append((topic_id, topic_names[topic_id], row_values))

    return rows, column_sums


def mean_and_ci(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Mean and 95% CI half-width (t-distribution, n = topics with data)."""
    if not vals:
        return (None, None)
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return (mean, None)
    variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
    se = math.sqrt(variance) / math.sqrt(n)
    return (mean, T_TABLE.get(n - 1, 1.96) * se)


def render_summary_table(column_sums: List[List[float]]) -> str:
    """Compact main-text table: one row per book class, cells are mean ± CI.

    Bolding is dropped here (unlike the per-topic table) since the CIs convey
    uncertainty better than a max-highlight; see the paired-difference report
    printed by main().
    """
    stats = [mean_and_ci(vals) for vals in column_sums]

    def cell(idx: int) -> str:
        mean, half = stats[idx]
        if mean is None:
            return "--"
        if half is None:
            return f"{mean:.2f}"
        return rf"{mean:.2f} $\pm$ {half:.2f}"

    lines: List[str] = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(" & " + " & ".join(CRITERION_HEADERS) + r" \\")
    lines.append(r"\midrule")
    for offset, label in enumerate(SUBCOLUMN_LABELS):
        cells = [cell(criterion * 3 + offset) for criterion in range(3)]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def render_per_topic_table(
    rows: List[Tuple[str, str, List[Optional[float]]]],
    column_sums: List[List[float]],
) -> str:
    lines: List[str] = []
    lines.append(r"\begin{tabular}{l|rrr|rrr|rrr}")
    lines.append(r"\toprule")
    lines.append(r"Topic & \multicolumn{3}{c|}{Interesting?} & \multicolumn{3}{c|}{Well-written?} & \multicolumn{3}{c}{Publishable?} \\")
    lines.append(" & " + " & ".join(SUBCOLUMN_LABELS * 3) + " \\\\")
    lines.append(r"\midrule")

    def format_group(values: List[Optional[float]]) -> List[str]:
        """Format a 3-value group, bolding the maximum present."""
        numeric = [v for v in values if v is not None]
        max_val = max(numeric) if numeric else None
        labels: List[str] = []
        for v in values:
            if v is None:
                labels.append("--")
            elif max_val is not None and v == max_val:
                labels.append(rf"\textbf{{{v:.2f}}}")
            else:
                labels.append(f"{v:.2f}")
        return labels

    for topic_id, topic_name, row_values in rows:
        interesting_labels = format_group(row_values[0:3])
        well_written_labels = format_group(row_values[3:6])
        publishable_labels = format_group(row_values[6:9])

        # topic_label = latex_escape(topic_id)
        MAX_LENGTH = 33
        # The truncation mark is a LaTeX command after escaping, not a raw "›": LuaLaTeX
        # hands U+203A to the T1 Courier font by code point, where it has no glyph.
        truncated_name = latex_escape(topic_name[:MAX_LENGTH])
        if len(topic_name) > MAX_LENGTH:
            truncated_name += r"\guilsinglright{}"

        # topic_label = rf"\href{{https://openalex.org/topics/{topic_id}}}{{\texttt{{\footnotesize {truncated_name}}}}}"
        topic_label = rf"\href{{https://openalex.org/topics/{topic_id}}}{{\texttt{{\scriptsize {truncated_name}}}}}"
        row = " ".join(
            [
                f"{topic_label} & {interesting_labels[0]} & {interesting_labels[1]} & {interesting_labels[2]} "
                f"& {well_written_labels[0]} & {well_written_labels[1]} & {well_written_labels[2]} "
                f"& {publishable_labels[0]} & {publishable_labels[1]} & {publishable_labels[2]} \\",
            ]
        )
        lines.append(row + "\\")

    # Average row: emitted as two LaTeX rows --- means on one, 95% CI
    # half-widths on the next --- so per-cell width stays narrow and the
    # decimal points align column-wise (both "4.10" and "$\pm$0.20" have
    # the decimal at position 2 from the right under \texttt{r}-columns).
    # Half-widths carry no "±" prefix; the row label "CI" plus the caption
    # together make it clear these are to be read as "mean ± value".
    def mean_and_ci_strs(vals: List[float]) -> Tuple[str, str]:
        mean, half = mean_and_ci(vals)
        if mean is None:
            return ("--", "")
        return (f"{mean:.2f}", "" if half is None else f"{half:.2f}")

    avg_pairs = [mean_and_ci_strs(vals) for vals in column_sums]
    avg_means_row = " & ".join(p[0] for p in avg_pairs)
    avg_cis_row = " & ".join(p[1] for p in avg_pairs)
    lines.append(r"\midrule")
    lines.append(f"Average & {avg_means_row} \\\\")
    lines.append(f"CI & {avg_cis_row} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-path",
        type=Path,
        default=Path("artifacts_scripts/survey/selected_books.csv"),
        help="Path to selected_books.csv",
    )
    parser.add_argument(
        "--survey-path",
        type=Path,
        default=Path("artifacts_scripts/survey/Survey_finalresults.csv"),
        help="Path to survey results CSV",
    )
    parser.add_argument(
        "--latex-path",
        type=Path,
        default=Path("paper/artifacts/selected_books_table.tex"),
        help="Where to write the compact main-text summary table (default: "
        "paper/artifacts/, where the paper \\inputs it from).",
    )
    parser.add_argument(
        "--per-topic-latex-path",
        type=Path,
        default=Path("paper/artifacts/selected_books_per_topic_table.tex"),
        help="Where to write the full per-topic appendix table.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Generate outputs even if mismatches are detected",
    )
    args = parser.parse_args()

    selected = load_selected_books(args.selected_path)
    survey_questions, survey_responses = load_survey(args.survey_path)

    if not args.skip_validation:
        validate_alignment(selected, survey_questions)

    means = compute_question_means(survey_questions, survey_responses)
    rows, column_sums = collect_cell_values(selected, means)

    provenance = latex_provenance(__spec__.name)
    args.latex_path.write_text(
        render_summary_table(column_sums) + provenance, encoding="utf-8"
    )
    args.per_topic_latex_path.write_text(
        render_per_topic_table(rows, column_sums) + provenance, encoding="utf-8"
    )

    print(f"Validated {len(selected)} selected books against {len(survey_questions)} survey questions.")
    print(f"Summary table written to {args.latex_path}")
    print(f"Per-topic table written to {args.per_topic_latex_path}")

    # Paired-difference statistics for §5.6: for each (rater, topic, criterion),
    # compute the differences (ours - high) and (ours - low) and report mean
    # ± 95% CI from a paired t-test (n = rater × topic pairs with both values).
    report_paired_differences(selected, survey_questions, survey_responses)


def report_paired_differences(
    selected: List[BookRecord],
    questions: List[SurveyQuestion],
    responses: List[Dict[str, str]],
) -> None:
    """Print paired-difference statistics for the §5.6 hybrid claim:
    `ours − ↑` and `ours − ↓` per criterion, with mean ± 95% CI.

    For each (rater, topic, criterion), we compute the score differences
    between the generated proposal and each of the two existing-book
    baselines. The paired-t framing accounts for both rater bias and
    per-topic difficulty automatically.
    """
    import math

    # Map book title -> (topic_id, system_label)
    by_topic_local: Dict[str, List[BookRecord]] = defaultdict(list)
    for rec in selected:
        by_topic_local[rec.topic_id].append(rec)
    title_to_system: Dict[str, Tuple[str, str]] = {}
    for topic_id, recs in by_topic_local.items():
        high, low, generated = choose_by_percentile(recs)
        if high:
            title_to_system[high.title] = (topic_id, "high")
        if low:
            title_to_system[low.title] = (topic_id, "low")
        if generated:
            title_to_system[generated.title] = (topic_id, "ours")

    # Index questions by (title, qtype) for lookup
    qkey_to_col: Dict[Tuple[str, QuestionType], str] = {
        (q.title, q.question_type): q.column_name for q in questions
    }

    criteria: List[QuestionType] = ["interesting", "well_written", "publishable"]
    # For each criterion, accumulate paired diffs across (rater, topic) pairs.
    diffs_ours_minus_high: Dict[QuestionType, List[float]] = {c: [] for c in criteria}
    diffs_ours_minus_low: Dict[QuestionType, List[float]] = {c: [] for c in criteria}

    for response in responses:
        for topic_id, recs in by_topic_local.items():
            high, low, generated = choose_by_percentile(recs)
            if not generated:
                continue
            for crit in criteria:
                gen_col = qkey_to_col.get((generated.title, crit))
                if gen_col is None:
                    continue
                gen_raw = response.get(gen_col, "").strip()
                if not gen_raw:
                    continue
                try:
                    gen_val = float(gen_raw)
                except ValueError:
                    continue
                if high:
                    hi_col = qkey_to_col.get((high.title, crit))
                    if hi_col is not None:
                        hi_raw = response.get(hi_col, "").strip()
                        try:
                            hi_val = float(hi_raw)
                            diffs_ours_minus_high[crit].append(gen_val - hi_val)
                        except ValueError:
                            pass
                if low:
                    lo_col = qkey_to_col.get((low.title, crit))
                    if lo_col is not None:
                        lo_raw = response.get(lo_col, "").strip()
                        try:
                            lo_val = float(lo_raw)
                            diffs_ours_minus_low[crit].append(gen_val - lo_val)
                        except ValueError:
                            pass

    def summarize(name: str, diffs: List[float]) -> str:
        n = len(diffs)
        if n < 2:
            return f"{name:<32s}  n={n} (insufficient data)"
        mean = sum(diffs) / n
        variance = sum((v - mean) ** 2 for v in diffs) / (n - 1)
        sd = math.sqrt(variance)
        se = sd / math.sqrt(n)
        # For n>30 (typical here), use z=1.96; otherwise use t (we expect n~150-180)
        crit = 1.96 if n > 30 else 2.0
        half = crit * se
        ci_lo, ci_hi = mean - half, mean + half
        contains_zero = ci_lo <= 0 <= ci_hi
        marker = "  (CI includes 0)" if contains_zero else "  (CI EXCLUDES 0)"
        return (f"{name:<32s}  mean={mean:+.3f}  95% CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]"
                f"  n={n}{marker}")

    print()
    print("Paired-difference statistics (per criterion, on the 1-7 Likert scale):")
    print("=" * 90)
    for crit in criteria:
        crit_label = crit.replace("_", "-")
        print(f"\n  {crit_label}:")
        print(f"    {summarize('ours − high-cited', diffs_ours_minus_high[crit])}")
        print(f"    {summarize('ours − low-cited',  diffs_ours_minus_low[crit])}")


if __name__ == "__main__":
    main()
