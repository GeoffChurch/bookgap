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
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

QuestionType = str

QUESTION_SLUGS = {
    "interesting": "interesting?",
    "well_written": "well-written?",
    "publishable": "worth publishing?",
}

EXPECTED_SUFFIX = "\n\n<br><br><b>Do you personally think this book is...</b>"

# Subcolumn labels used across the three question groups (typeset monospaced).
SUBCOLUMN_LABELS = (r"\texttt{↑}", r"\texttt{↓}", r"\texttt{ours}")

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


def render_latex_table(
    selected: List[BookRecord],
    means: Dict[Tuple[str, QuestionType], Optional[float]],
) -> str:
    by_topic: Dict[str, List[BookRecord]] = defaultdict(list)
    topic_names: Dict[str, str] = {}
    for rec in selected:
        by_topic[rec.topic_id].append(rec)
        topic_names[rec.topic_id] = rec.topic_name

    lines: List[str] = []
    lines.append(r"\begin{tabular}{l|rrr|rrr|rrr}")
    lines.append(r"\toprule")
    lines.append(r"Topic & \multicolumn{3}{c|}{Interesting?} & \multicolumn{3}{c|}{Well-written?} & \multicolumn{3}{c}{Publishable?} \\")
    lines.append(" & " + " & ".join(SUBCOLUMN_LABELS * 3) + " \\\\")
    lines.append(r"\midrule")

    column_order = [
        ("interesting", "high"),
        ("interesting", "low"),
        ("interesting", "generated"),
        ("well_written", "high"),
        ("well_written", "low"),
        ("well_written", "generated"),
        ("publishable", "high"),
        ("publishable", "low"),
        ("publishable", "generated"),
    ]
    column_sums: List[List[float]] = [[] for _ in column_order]

    def value(rec: Optional[BookRecord], qtype: QuestionType) -> Optional[float]:
        if not rec:
            return None
        return means.get((rec.title, qtype))

    def format_group(values: List[Optional[float]]) -> List[str]:
        """Format a 3-value group, bolding the maximum present."""
        numeric = [v for v in values if v is not None]
        max_val = max(numeric) if numeric else None
        labels: List[str] = []
        for v in values:
            if v is None:
                labels.append("--")
            elif max_val is not None and v == max_val:
                labels.append(rf"\textbf{{{v:.1f}}}")
            else:
                labels.append(f"{v:.1f}")
        return labels

    for topic_id in sorted(by_topic.keys()):
        records = by_topic[topic_id]
        high, low, generated = choose_by_percentile(records)

        row_values = [
            value(high, "interesting"),
            value(low, "interesting"),
            value(generated, "interesting"),
            value(high, "well_written"),
            value(low, "well_written"),
            value(generated, "well_written"),
            value(high, "publishable"),
            value(low, "publishable"),
            value(generated, "publishable"),
        ]
        for idx, val in enumerate(row_values):
            if val is not None:
                column_sums[idx].append(val)

        interesting_labels = format_group(row_values[0:3])
        well_written_labels = format_group(row_values[3:6])
        publishable_labels = format_group(row_values[6:9])

        # topic_label = latex_escape(topic_id)
        MAX_LENGTH = 35
        truncated_name = topic_names[topic_id][:MAX_LENGTH]
        if len(topic_names[topic_id]) > MAX_LENGTH:
            truncated_name += "…"
        
        topic_label = rf"\texttt{{{latex_escape(truncated_name)}}}"
        topic_label = rf"\href{{https://openalex.org/topics/{topic_id}}}{{{topic_label}}}"
        row = " ".join(
            [
                f"{topic_label} & {interesting_labels[0]} & {interesting_labels[1]} & {interesting_labels[2]} "
                f"& {well_written_labels[0]} & {well_written_labels[1]} & {well_written_labels[2]} "
                f"& {publishable_labels[0]} & {publishable_labels[1]} & {publishable_labels[2]} \\",
            ]
        )
        lines.append(row + "\\")

    def format_avg(vals: List[float]) -> str:
        if not vals:
            return "--"
        return f"{(sum(vals) / len(vals)):.1f}"

    avg_numeric = [format_avg(vals) for vals in column_sums]
    def parse_avg(val: str) -> Optional[float]:
        try:
            return float(val)
        except ValueError:
            return None
    avg_labels = (
        format_group([parse_avg(v) for v in avg_numeric[0:3]])
        + format_group([parse_avg(v) for v in avg_numeric[3:6]])
        + format_group([parse_avg(v) for v in avg_numeric[6:9]])
    )
    avg_row = " & ".join(avg_labels)
    lines.append(r"\midrule")
    lines.append(f"Average & {avg_row} \\\\")

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
        default=Path("artifacts_scripts/survey/Survey_+Textbook+Proposal+Evaluation_December+19,+2025_10.55.csv"),
        help="Path to survey results CSV",
    )
    parser.add_argument(
        "--latex-path",
        type=Path,
        default=Path("artifacts_scripts/survey/selected_books_table.tex"),
        help="Where to write the generated LaTeX table",
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
    latex = render_latex_table(selected, means)
    args.latex_path.write_text(latex, encoding="utf-8")

    print(f"Validated {len(selected)} selected books against {len(survey_questions)} survey questions.")
    print(f"LaTeX table written to {args.latex_path}")


if __name__ == "__main__":
    main()
