import csv
from dataclasses import dataclass
from datetime import date, datetime
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
import re

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from bookgap.cloud import Cloud
from bookgap.configs.config_store import register_configs
from bookgap.subcloud import get_subclouds_for_topic
from bookgap.util import setup_logging

# Register configs
register_configs()

LOGGER = logging.getLogger(__name__)

# GAPS_CSV_PATH = Path("/home/gchurchill/src/bookgap/outputs/2025-11-09/09-48-25/gaps_analysis_results.csv") # w/ ToCs
GAPS_CSV_PATH = Path("/home/gchurchill/src/bookgap/outputs/2025-12-04/22-56-46/gaps_analysis_results.csv") # w/o ToCs

NUM_TOPICS = 8

BLACKLIST_TOPIC_IDS = {
    "T12157",  # Geochemistry and Geologic Mapping
    "T13623",  # Linguistic Studies and Language Acquisition
    "T13398",  # Data Analysis with R
    "T13650",  # Computational Physics and Python Applications
    
    # Sticking more with topics related to NLP/CV/ML, we blacklist the following:
    "T10126",  # Logic, programming, and type systems
    "T11010",  # Logic, Reasoning, and Knowledge
    "T10639",  # Advanced Software Engineering Methodologies"

    "T13734",  # Advanced Computational Techniques and Applications (only one valid book...)
    "T10906",  # AI-based Problem Solving and Planning (only one valid book...)
}

FORCELIST_TOPIC_IDS = {
    "T11902",  # Intelligent Tutoring Systems and Adaptive Learning: 224 books
}

BLACKLIST_WORK_IDS = {
    "https://openalex.org/W2916814366",  # T11574 -- Artificial Intelligence in Games -- Playing Smart : On Games, Intelligence, and Artificial Intelligence (mentions author)
    "https://openalex.org/W3145031344",  # T11574 -- Artificial Intelligence in Games -- Computer Chess (strange abstract)
    "https://openalex.org/W2533512318",  # T11574 -- Artificial Intelligence in Games -- Game Mechanics: Advanced Game Design (mentions author)
    "https://openalex.org/W1487104801",  # T11574 -- Artificial Intelligence in Games -- Artificial Intelligence for Games (mentions author)
    "https://openalex.org/W99317934",    # T10215 -- Semantic Web and Ontologies -- Terminological Ontologies: Design, Management and Practical Applications (name drop)
    "https://openalex.org/W1128038141",  # T10181 -- Natural Language Processing Techniques -- Scientific and Technical Translation Explained: A Nuts and Bolts Guide for Beginners (ToC-like abstract)
    "https://openalex.org/W1966696206",  # T13734 -- Advanced Computational Techniques and Applications -- Dewey Decimal Classification and Relative index
    "https://openalex.org/W1579148054",  # T13734 -- Advanced Computational Techniques and Applications -- STL tutorial and reference guide (junk abstract)
    "https://openalex.org/W2966503377",  # T13734 -- Advanced Computational Techniques and Applications -- Analysis and Design of Information Systems (junk abstract)
    "https://openalex.org/W1414451058",  # T13734 -- Advanced Computational Techniques and Applications -- Autonomous Mobile Robots (list-like abstract)
    "https://openalex.org/W160461043",   # T13734 -- Advanced Computational Techniques and Applications -- Explanation and Cognition (lots of names listed)
    "https://openalex.org/W2494449340",  # T13734 -- Advanced Computational Techniques and Applications -- Comprehensive Glossary of Telecom Abbreviations and Acronyms (not textbook-like)
    "https://openalex.org/W4299591165",  # T10906 -- AI-based Problem Solving and Planning -- A Concise Introduction to Models and Methods for Automated Planning (truncated abstract)
    "https://openalex.org/W1571310641",  # T10906 -- AI-based Problem Solving and Planning -- Understanding Planning Tasks (ToC-like abstract)
    "https://openalex.org/W610756163",   # T10906 -- AI-based Problem Solving and Planning -- Fuzzy expert systems and fuzzy reasoning (ToC-like abstract)
    "https://openalex.org/W1595507273",  # T10906 -- AI-based Problem Solving and Planning -- Foundations of real-world intelligence (mentions authors)
    "https://openalex.org/W140777655",   # T11269 -- Algorithms and Data Compression -- Computational Methods of Feature Selection (very long abstract)
    "https://openalex.org/W1506907352",  # T11269 -- Algorithms and Data Compression -- Introduction to the design & analysis of algorithms (ToC-like abstract)
    "https://openalex.org/W562026345",   # T10820 -- Fuzzy Logic and Control Systems -- Fuzzy Control Systems Design and Analysis: A Linear Matrix Inequality Approach (authors mentioned)
    "https://openalex.org/W2106192381",  # T10126 -- Logic, programming, and type systems -- Isabelle/HOL: A Proof Assistant for Higher-Order Logic (manual blacklist per user)
    "https://openalex.org/W4312811117",  # T10181 -- Natural Language Processing Techniques -- Bayesian Analysis in Natural Language Processing (truncated abstract)
    "https://openalex.org/W2894218541",  # T10181 -- Natural Language Processing Techniques -- Quality Estimation for Machine Translation (truncated abstract)
    "https://openalex.org/W1605394365",  # T10181 -- Natural Language Processing Techniques -- The Handbook of Japanese Linguistics (ToC-like abstract)
    "https://openalex.org/W4243052213",  # T10181 -- Natural Language Processing Techniques -- Spoken Corpus Linguistics (authors named)
    "https://openalex.org/W3168499366",  # T10181 -- Natural Language Processing Techniques -- Verb Valency Patterns (covers results from a study)
    "https://openalex.org/W2574053021",  # T10181 -- Natural Language Processing Techniques -- EVALITA. Evaluation of NLP and Speech Tools for Italian (evaluation campaign)
    "https://openalex.org/W4298445375",  # T10181 -- Natural Language Processing Techniques -- Connectionist Approaches to Natural Language Processing (references earlier edition)
    "https://openalex.org/W609663673",   # T10320 -- Neural Networks and Applications -- Handbook of time series analysis : recent theoretical developments and applications (very long abstract)
    "https://openalex.org/W1545139845",  # T10320 -- Neural Networks and Applications -- Neural-Symbolic Learning Systems: Foundations and Applications (ToC-like abstract)
    "https://openalex.org/W1718459485",  # T10215 -- Semantic Web and Ontologies -- Semantic Web for the Working Ontologist (obvious real book)
    "https://openalex.org/W2268238446",  # T10215 -- Semantic Web and Ontologies -- Instant Apache Stanbol (user manual)
    "https://openalex.org/W2611894836",  # T10215 -- Semantic Web and Ontologies -- Ontology Matching (authors mentioned)
}

# Configuration
MORE_PCT = 0.95  # Target percentile for "highly cited" book
LESS_PCT = 0.30  # Target percentile for "less cited" book
KERNEL_BANDWIDTH_YEARS = 2.0  # Gaussian kernel bandwidth in years
N_BOOTSTRAP = 500  # Number of bootstrap samples for CI
MIN_CITATIONS = 2  # Minimum citations to qualify
MIN_YEAR = 2000  # Minimum publication year

# Either of the following length limits can be set to None to disable that filter
MIN_ABSTRACT_LENGTH = 1300
MAX_ABSTRACT_LENGTH = 3500


@dataclass
class BookMetrics:
    """Metrics for a book computed with kernel-weighted temporal smoothing."""
    book: Dict[str, Any]
    decimal_year: float
    citations: int
    kernel_percentile: float
    percentile_ci_low: float
    percentile_ci_high: float
    n_eff: float
    raw_year_n: int
    raw_year_percentile: float


def parse_date(date_str: str) -> date | None:
    """Parse YYYY-MM-DD string to date object."""
    if not date_str or len(date_str) < 10:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def date_to_decimal_year(d: date) -> float:
    """Convert date to decimal year (e.g., 2005.5 for mid-2005)."""
    year_start = date(d.year, 1, 1)
    year_end = date(d.year + 1, 1, 1)
    year_fraction = (d - year_start).days / (year_end - year_start).days
    return d.year + year_fraction


def has_toc_like_abstract(book: Dict[str, Any]) -> bool:
    """
    Return True if the book's abstract looks like a table-of-contents, prefatory text,
    or similar listing that is unsuitable for survey presentation.
    Also logs the abstract when detected so maintainers can examine or tune the heuristics.
    """
    abstract_text = book.get("abstract", "") or ""
    lower_abs = abstract_text.lower()

    toc_phrases = (
        "table of contents",
        "list of contributors",
        "list of figures",
        "list of tables",
        # 'preface' can indicate ToC but also appears in publisher blurbs; comment out to avoid false positives
        # "preface",
        "contents:",
        "1. introduction.",
        "1. preface ",
        "this thesis",
        "from the publisher:",
    )
    toc_phrases = [phrase.lower() for phrase in toc_phrases]
    for phrase in toc_phrases:
        if phrase in lower_abs:
            LOGGER.warning(f"Excluding book {book.get('id') or book.get('display_name', 'unknown')} due to ToC-like abstract (phrase: '{phrase}')")
            LOGGER.info(f"Abstract: {abstract_text}")
            return True

    # If the abstract mentions 'chapter' many times, it's likely a TOC or contents listing
    chapter_count = lower_abs.count('chapter')
    if chapter_count >= 7:
        LOGGER.warning(f"Excluding book {book.get('id') or book.get('display_name', 'unknown')} due to ToC-like abstract ({chapter_count} 'chapter' mentions)")
        LOGGER.info(f"Abstract: {abstract_text}")
        return True

    # Detect multiple numbered list lines at the start of lines (1. Introduction, 2. ...)
    numbered_entries = len(re.findall(r'^\s*\d+\s*[\.)]\s+', abstract_text, flags=re.MULTILINE))
    # Similarly, require at least 4 numbered entries for a strong ToC signal
    if numbered_entries >= 4:
        LOGGER.warning(f"Excluding book {book.get('id') or book.get('display_name', 'unknown')} due to ToC-like abstract (numbered list lines: {numbered_entries})")
        LOGGER.info(f"Abstract: {abstract_text}")
        return True

    return False


def gaussian_kernel(delta_years: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian kernel weight based on time difference."""
    return np.exp(-0.5 * (delta_years / bandwidth) ** 2)


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute effective sample size from weights: (sum w)^2 / sum(w^2)."""
    w_sum = weights.sum()
    if w_sum == 0:
        return 0.0
    return (w_sum ** 2) / (weights ** 2).sum()


def compute_kernel_percentile_with_ci(
    book_citations: int,
    book_decimal_year: float,
    all_citations: np.ndarray,
    all_decimal_years: np.ndarray,
    bandwidth: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """
    Compute kernel-weighted percentile with bootstrap confidence interval.
    
    Returns: (percentile, ci_low, ci_high, n_eff)
    """
    # Compute kernel weights
    weights = gaussian_kernel(all_decimal_years - book_decimal_year, bandwidth)
    weight_sum = weights.sum()
    if weight_sum == 0.0:
        # Degenerate case: no effective neighbors; return neutral stats
        return 50.0, 50.0, 50.0, 0.0

    # Point estimate: weighted percentile
    leq_mask = all_citations <= book_citations
    weighted_leq = float((weights * leq_mask).sum()) / weight_sum
    percentile = weighted_leq * 100.0
    
    # Effective sample size
    n_eff = effective_sample_size(weights)
    
    # Bootstrap CI: binomial on success prob weighted_leq with sample size len(all_citations)
    # This is equivalent to the weighted-resample-and-count procedure but vectorized.
    n_draws = len(all_citations)
    bootstrap_successes = rng.binomial(n=n_draws, p=weighted_leq, size=n_bootstrap)
    bootstrap_pctls = (bootstrap_successes / n_draws) * 100.0
    ci_low, ci_high = np.percentile(bootstrap_pctls, [2.5, 97.5])
    
    return percentile, ci_low, ci_high, n_eff


def read_top_topics_from_csv(csv_path: Path) -> List[str]:
    """Read gaps_analysis_results.csv and return top NUM_TOPICS topic_ids by num_books,
    excluding any topics in BLACKLIST_TOPIC_IDS before sorting/selecting,
    and always including topics from FORCELIST_TOPIC_IDS.
    """
    # Validate configuration
    overlap = FORCELIST_TOPIC_IDS & BLACKLIST_TOPIC_IDS
    if overlap:
        raise ValueError(f"Topic forcelist overlaps blacklist: {sorted(overlap)}")
    if len(FORCELIST_TOPIC_IDS) > NUM_TOPICS:
        raise ValueError(
            f"Forcelist size ({len(FORCELIST_TOPIC_IDS)}) exceeds NUM_TOPICS ({NUM_TOPICS}); increase NUM_TOPICS or shrink the forcelist"
        )

    forcelisted = []
    other_topics = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                topic_id = row["topic_id"]
                topic_name = row.get("topic_name", "")
                num_books = int(row["num_books"]) if row.get("num_books") else 0
                if topic_id in BLACKLIST_TOPIC_IDS:
                    continue
                if topic_id in FORCELIST_TOPIC_IDS:
                    forcelisted.append((topic_id, topic_name, num_books))
                else:
                    other_topics.append((topic_id, topic_name, num_books))
            except Exception as e:
                LOGGER.warning(f"Skipping CSV row due to parse error: {e}")
                continue
    
    # Sort other topics by num_books and take top (NUM_TOPICS - len(forcelisted))
    other_topics.sort(key=lambda x: x[2], reverse=True)
    num_other = max(0, NUM_TOPICS - len(forcelisted))
    selected_topics = forcelisted + other_topics[:num_other]

    # Sort selected (for CSV order) and full list (for logging) by num_books desc
    selected_topics.sort(key=lambda x: x[2], reverse=True)
    topics = sorted(forcelisted + other_topics, key=lambda x: x[2], reverse=True)
    
    topic_str = "\n".join([f"{t[0]} ({t[1]}): {t[2]} books" for t in topics])
    LOGGER.info(f"Available topics after blacklist:\n{topic_str}")
    LOGGER.info(f"Forcelisted topics: {[t[0] for t in forcelisted]}")
    LOGGER.info(f"Selected topics: {[t[0] for t in selected_topics]}")
    return [t[0] for t in selected_topics]


def read_topic_names(csv_path: Path) -> Dict[str, str]:
    """Map topic_id -> topic_name from gaps_analysis_results.csv."""
    mapping: Dict[str, str] = {}
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("topic_id")
            name = row.get("topic_name")
            if tid and name:
                mapping[tid] = name
    return mapping


def read_proposals_from_csv(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Map topic_id -> generated proposal {title, abstract} from gaps_analysis_results.csv."""
    proposals: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("topic_id")
            title = row.get("proposal_title")
            abstract = row.get("proposal_abstract")
            if tid and title:
                proposals[tid] = {
                    "title": title,
                    "abstract": abstract or "",
                    "source": "generated",
                    "topic_id": tid,
                }
    return proposals


def get_citation_counts(works: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compute incoming citation counts within the cloud."""
    counts = Counter()
    cloud_work_ids = set(w["id"] for w in works)
    
    for work in works:
        refs = work.get("referenced_works", [])
        for ref in refs:
            if ref in cloud_work_ids:
                counts[ref] += 1
    return counts


def is_statistically_valid(book: Dict[str, Any], pub_date: date | None) -> bool:
    """
    Check if a book should be included in the statistical distribution.
    
    These are hard requirements for meaningful citation analysis:
    - Must have a parseable publication date
    - Must be published after MIN_YEAR (older works have different citation dynamics)
    """
    if pub_date is None:
        return False
    if pub_date.year < MIN_YEAR:
        return False
    return True


def is_selectable_for_output(
    book: Dict[str, Any],
    citations: int,
) -> bool:
    """
    Check if a book can be selected for the output CSV.
    
    These filters don't affect statistical validity - the books still contribute
    to percentile calculations - but they make the book unsuitable for the survey:
    - Blacklisted works (user manuals, truncated abstracts, etc.)
    - Too few citations to be meaningfully compared
    - Missing abstract text
    """
    if book["id"] in BLACKLIST_WORK_IDS:
        return False
    if citations < MIN_CITATIONS:
        return False
    if "[[MISSING]]" in book.get("abstract", ""):
        return False
    # Exclude very short abstracts which are unsuitable for survey presentation
    if MIN_ABSTRACT_LENGTH is not None and len(book.get("abstract", "")) < MIN_ABSTRACT_LENGTH:
        # LOGGER.warning(f"Excluding book {book.get('id') or book.get('display_name', 'unknown')} due to short abstract ({len(book.get('abstract', ''))} chars)")
        # LOGGER.info(f"Abstract: {book.get('abstract', '')}")
        return False
    # Exclude very long abstracts which are unsuitable for survey presentation
    if MAX_ABSTRACT_LENGTH is not None and len(book.get("abstract", "")) > MAX_ABSTRACT_LENGTH:
        # LOGGER.warning(f"Excluding book {book.get('id') or book.get('display_name', 'unknown')} due to long abstract ({len(book.get('abstract', ''))} chars)")
        # LOGGER.info(f"Abstract: {book.get('abstract', '')}")
        return False
    if has_toc_like_abstract(book):
        return False
    return True


def select_percentile_books_for_topic(
    topic_id: str,
    topic_books: List[Dict[str, Any]],
    citation_counts: Dict[str, int],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """
    For a given topic's books, pick one at the MORE_PCT percentile and one at the LESS_PCT percentile
    based on kernel-weighted temporal percentiles with bootstrap CIs.
    
    Also exports all statistically valid books to a CSV file for downstream analysis.
    
    Returns list of two dicts with title/abstract/source plus metrics.
    """
    # 1. Build list of statistically valid books (for percentile computation)
    #    and track which are also selectable for output
    valid_books = []
    for b in topic_books:
        citations = citation_counts.get(b["id"], 0)
        pub_date = parse_date(b.get("publication_date", ""))
        
        if not is_statistically_valid(b, pub_date):
            continue
        
        # Type narrowing: is_statistically_valid guarantees pub_date is not None
        assert pub_date is not None
            
        valid_books.append({
            "book": b,
            "citations": citations,
            "pub_date": pub_date,
            "decimal_year": date_to_decimal_year(pub_date),
            "is_selectable": is_selectable_for_output(b, citations),
        })

    if not valid_books:
        raise ValueError(f"Topic has no statistically valid books")

    # 2. Prepare arrays for kernel computation (ALL valid books for accurate percentiles)
    all_citations = np.array([vb["citations"] for vb in valid_books])
    all_decimal_years = np.array([vb["decimal_year"] for vb in valid_books])
    
    # 3. Compute metrics for each book
    books_with_metrics: List[tuple[BookMetrics, bool]] = []
    
    for vb in valid_books:
        book = vb["book"]
        citations = vb["citations"]
        decimal_year = vb["decimal_year"]
        pub_year = vb["pub_date"].year
        is_selectable = vb["is_selectable"]
        
        # Compute kernel-weighted percentile with CI
        kernel_pctl, ci_low, ci_high, n_eff = compute_kernel_percentile_with_ci(
            book_citations=citations,
            book_decimal_year=decimal_year,
            all_citations=all_citations,
            all_decimal_years=all_decimal_years,
            bandwidth=KERNEL_BANDWIDTH_YEARS,
            n_bootstrap=N_BOOTSTRAP,
            rng=rng,
        )
        
        # Compute raw year-only metrics for reference
        year_mask = (all_decimal_years >= pub_year) & (all_decimal_years < pub_year + 1)
        raw_year_n = year_mask.sum()
        if raw_year_n > 0:
            year_citations = all_citations[year_mask]
            raw_year_percentile = (year_citations <= citations).mean() * 100.0
        else:
            raw_year_percentile = 50.0  # Fallback
        
        books_with_metrics.append((
            BookMetrics(
                book=book,
                decimal_year=decimal_year,
                citations=citations,
                kernel_percentile=kernel_pctl,
                percentile_ci_low=ci_low,
                percentile_ci_high=ci_high,
                n_eff=n_eff,
                raw_year_n=raw_year_n,
                raw_year_percentile=raw_year_percentile,
            ),
            is_selectable,
        ))

    # 4. Select books based on their kernel percentile rank (only from selectable books)
    selectable_books = [bm for bm, selectable in books_with_metrics if selectable]
    
    if not selectable_books:
        raise ValueError(f"Topic has no selectable books after output filtering")
    
    # Export statistically valid books for analysis
    valid_books_for_export = []
    for vb in valid_books:
        book = vb["book"]
        citations = vb["citations"]
        abstract_len = len(book.get("abstract", ""))
        # Track if this book made it through output selection
        is_selectable = any(bm.book["id"] == book["id"] for bm, _ in books_with_metrics if _)
        valid_books_for_export.append({
            "topic_id": topic_id,
            "work_id": book.get("id", ""),
            "title": book.get("display_name", ""),
            "abstract_length": abstract_len,
            "citations": citations,
            "publication_date": book.get("publication_date", ""),
            "is_selectable": is_selectable,
        })
    
    # Write to CSV
    out_path = Path(__file__).parent / f"valid_books_{topic_id}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["topic_id", "work_id", "title", "abstract_length", "citations", "publication_date", "is_selectable"]
        )
        writer.writeheader()
        writer.writerows(valid_books_for_export)
    LOGGER.info(f"Exported {len(valid_books_for_export)} valid books for topic {topic_id} to {out_path}")
    
    selectable_books.sort(key=lambda x: x.kernel_percentile)
    
    N = len(selectable_books)
    k_less = max(0, min(N - 1, math.ceil(LESS_PCT * N) - 1))
    k_more = max(0, min(N - 1, math.ceil(MORE_PCT * N) - 1))

    if k_less == k_more:
        k_less = 0
        k_more = N - 1

    metrics_less = selectable_books[k_less]
    metrics_more = selectable_books[k_more]
    
    # Ensure we selected distinct books
    if metrics_less.book["id"] == metrics_more.book["id"]:
        if N < 2:
            raise ValueError(f"Topic {topic_id} has fewer than 2 selectable books; cannot select distinct high and low percentile books")
        else:
            raise ValueError(
                f"Topic {topic_id} selected the same book for both high and low percentiles "
                f"(book_id={metrics_less.book['id']}, title={metrics_less.book.get('display_name', 'unknown')}). "
                f"This should not happen with {N} selectable books. Check filtering logic."
            )

    # Helper to compute outbound citations
    def outbound_refs(book: Dict[str, Any]) -> int:
        return len(book.get("referenced_works", []) or [])

    def build_result(metrics: BookMetrics, source: str) -> Dict[str, Any]:
        return {
            "title": metrics.book.get("display_name", "No Title"),
            "abstract": metrics.book.get("abstract", "No Abstract"),
            "source": source,
            "book_id": metrics.book.get("id"),
            "publication_date": metrics.book.get("publication_date", ""),
            "incoming_citations": metrics.citations,
            "outgoing_citations": outbound_refs(metrics.book),
            "kernel_percentile": metrics.kernel_percentile,
            "percentile_ci_low": metrics.percentile_ci_low,
            "percentile_ci_high": metrics.percentile_ci_high,
            "n_eff": metrics.n_eff,
            "raw_year_n": metrics.raw_year_n,
            "raw_year_percentile": metrics.raw_year_percentile,
            "bandwidth_years": KERNEL_BANDWIDTH_YEARS,
        }

    return [
        build_result(metrics_more, "high_percentile"),
        build_result(metrics_less, "low_percentile"),
    ]


def clean_book(book: Dict[str, Any]) -> str | None:
    """
    Modifies the book in-place.
    Returns the type of transformation applied, or None if no transformation.
    """
    # If the abstract contains a ToC, clip everything from that point onward.
    abstract: str = book["abstract"]
    abstract = abstract.strip()
    match = re.search(r"\s*table of contents", abstract, re.IGNORECASE)
    if match:
        book["abstract"] = abstract[:match.start()]
        return "toc_removed"
    return None


@hydra.main(version_base=None, config_name="standard")
def main(cfg: DictConfig):
    setup_logging(log_name="select_survey_books", log_level="INFO")
    
    # Initialize RNG for reproducibility
    rng = np.random.default_rng(42)
    
    # 1. Load Cloud
    LOGGER.info("Instantiating cloud...")
    cloud: Cloud = instantiate(cfg.cloud)

    LOGGER.info(f"Cloud: {cloud.name()}")
    
    # 2. Determine top NUM_TOPICS topics by num_books from CSV
    if not GAPS_CSV_PATH.exists():
        LOGGER.error(f"Topics CSV not found: {GAPS_CSV_PATH}")
        return
    topics = read_top_topics_from_csv(GAPS_CSV_PATH)
    LOGGER.info(f"Top {NUM_TOPICS} topics: {topics}")
    topic_names = read_topic_names(GAPS_CSV_PATH)

    # Load generated proposals mapped by topic_id from CSV
    generated_by_topic = read_proposals_from_csv(GAPS_CSV_PATH)
    LOGGER.info(f"Loaded generated proposals for {len(generated_by_topic)} topics from CSV")

    # 3. Compute citation counts across all works once
    all_works = cloud.works()
    citation_counts = get_citation_counts(all_works)
    
    # 4. For each topic, pick three books: high percentile, low percentile, generated
    csv_rows: List[Dict[str, Any]] = []
    for topic_id in topics:
        LOGGER.info(f"Selecting percentile books for topic {topic_id}...")
        books_subcloud, _ = get_subclouds_for_topic(cloud, topic_id)
        topic_books = books_subcloud.works()
        LOGGER.info(f"  Topic {topic_id} has {len(topic_books)} books")

        # Track transformations for each book
        book_transformations: Dict[str, str | None] = {}
        for book in topic_books:
            transformation = clean_book(book)
            if transformation:
                book_transformations[book["id"]] = transformation

        # Select percentile-based books
        percentile_books = select_percentile_books_for_topic(topic_id, topic_books, citation_counts, rng)

        # Prepare CSV rows for the two existing books
        for pb in percentile_books:
            book_id = pb.get("book_id", "")
            len_abstract = len(pb.get("abstract", ""))
            csv_rows.append({
                "topic_id": topic_id,
                "topic_name": topic_names.get(topic_id, ""),
                "book_id": book_id,
                "publication_date": pb.get("publication_date", ""),
                "title": pb.get("title", ""),
                "abstract": pb.get("abstract", ""),
                "len_abstract": len_abstract,
                "incoming_citations": pb.get("incoming_citations", 0),
                "outgoing_citations": pb.get("outgoing_citations", 0),
                "kernel_percentile": pb.get("kernel_percentile", 0.0),
                "percentile_ci_low": pb.get("percentile_ci_low", 0.0),
                "percentile_ci_high": pb.get("percentile_ci_high", 0.0),
                "n_eff": pb.get("n_eff", 0.0),
                "raw_year_n": pb.get("raw_year_n", 0),
                "raw_year_percentile": pb.get("raw_year_percentile", 0.0),
                "bandwidth_years": pb.get("bandwidth_years", KERNEL_BANDWIDTH_YEARS),
                "transformation": book_transformations.get(book_id),
            })

        # Include the generated book for this topic if available
        gen = generated_by_topic.get(topic_id)
        if gen:
            gen_abs = gen.get("abstract", "No Abstract")
            csv_rows.append({
                "topic_id": topic_id,
                "topic_name": topic_names.get(topic_id, ""),
                "book_id": "",
                "publication_date": "",
                "title": gen.get("title", "No Title"),
                "abstract": gen_abs,
                "len_abstract": len(gen_abs),
                "incoming_citations": None,
                "outgoing_citations": None,
                "kernel_percentile": None,
                "percentile_ci_low": None,
                "percentile_ci_high": None,
                "n_eff": None,
                "raw_year_n": None,
                "raw_year_percentile": None,
                "bandwidth_years": None,
                "transformation": None,
            })
        else:
            raise ValueError(f"No generated proposal found for topic {topic_id} in CSV")
    
    # 5. Write to CSV
    csv_out = Path(__file__).parent / "selected_books.csv"
    
    with csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "topic_id",
                "topic_name",
                "book_id",
                "publication_date",
                "incoming_citations",
                "outgoing_citations",
                "kernel_percentile",
                "percentile_ci_low",
                "percentile_ci_high",
                "n_eff",
                "raw_year_n",
                "raw_year_percentile",
                "bandwidth_years",
                "transformation",
                "len_abstract",
                "title",
                "abstract",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    
    # Log abstracts sorted by length (ascending)
    sorted_by_len = sorted(csv_rows, key=lambda r: r.get("len_abstract", 0))
    LOGGER.info("Abstract lengths (ascending):")
    for row in sorted_by_len:
        LOGGER.info(
            f"  len={row.get('len_abstract', 0):>4} | topic={row.get('topic_id', '')} | "
            f"book_id={row.get('book_id', '') or '[generated]'} | title={row.get('title', '')}"
        )

    LOGGER.info(f"Selected books CSV written to {csv_out}")


if __name__ == "__main__":
    main()
