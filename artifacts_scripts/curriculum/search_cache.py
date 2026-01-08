#!/usr/bin/env python
"""
Search the OpenAlex API data cache for works matching keywords.

This script searches across all cached JSON files in parallel to find works
matching specified keywords in their titles or abstracts.

Usage:
    python -m artifacts_scripts.curriculum.search_cache "tree of thoughts"
    python -m artifacts_scripts.curriculum.search_cache "attention is all you need" --max-results 5
    python -m artifacts_scripts.curriculum.search_cache "transformer" --field title

TODO: could fuzzy search the API for missing works:
https://api.openalex.org/works?filter=title_and_abstract.search:%22lora%22,primary_topic.subfield.id:1702,type:article&select=id,created_date,title,citation_normalized_percentile&sort=citation_normalized_percentile.value:desc&page=2
(API can be searched with the `openalex` package)
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SearchResult:
    """A work matching the search query."""
    openalex_id: str
    display_name: str
    publication_date: Optional[str]
    work_type: str
    abstract_snippet: str
    match_score: float  # Higher = more keyword matches


def reconstruct_abstract(abstract_inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not abstract_inverted_index:
        return ""
    
    # Build list of (position, word) tuples
    word_positions = []
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    
    # Sort by position and join
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


def search_file(
    file_path: Path,
    keywords: list[str],
    field: str,
    case_sensitive: bool,
) -> list[SearchResult]:
    """Search a single JSON file for matching works.
    
    Args:
        file_path: Path to the JSON page file
        keywords: List of keywords to search for
        field: Which field to search ('title', 'abstract', 'both')
        case_sensitive: Whether search is case-sensitive
        
    Returns:
        List of SearchResult objects for matching works
    """
    results = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            works = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return results
    
    for work in works:
        # Get searchable text
        title = work.get("display_name", "") or ""
        
        # Handle abstract - may be string or inverted index
        abstract_raw = work.get("abstract", "")
        if isinstance(abstract_raw, dict):
            abstract = reconstruct_abstract(abstract_raw)
        else:
            abstract = abstract_raw or ""
        
        # Prepare text based on field
        if field == "title":
            search_text = title
        elif field == "abstract":
            search_text = abstract
        else:  # both
            search_text = f"{title} {abstract}"
        
        if not case_sensitive:
            search_text_lower = search_text.lower()
            keywords_to_match = [k.lower() for k in keywords]
        else:
            search_text_lower = search_text
            keywords_to_match = keywords
        
        # Count matches
        match_count = sum(1 for kw in keywords_to_match if kw in search_text_lower)
        
        if match_count > 0:
            # Extract OpenAlex ID
            openalex_id = work.get("id", "")
            if openalex_id.startswith("https://openalex.org/"):
                openalex_id = openalex_id.split("/")[-1]
            
            # Create abstract snippet
            abstract_snippet = abstract[:200] + "..." if len(abstract) > 200 else abstract
            
            results.append(SearchResult(
                openalex_id=openalex_id,
                display_name=title,
                publication_date=work.get("publication_date"),
                work_type=work.get("type", "unknown"),
                abstract_snippet=abstract_snippet,
                match_score=match_count / len(keywords),
            ))
    
    return results


def search_cache(
    cache_dir: Path,
    keywords: list[str],
    field: str = "both",
    case_sensitive: bool = False,
    max_results: int = 20,
    n_workers: Optional[int] = None,
) -> list[SearchResult]:
    """Search all cached JSON files for works matching keywords.
    
    Args:
        cache_dir: Directory containing page_*.json files (searched recursively)
        keywords: List of keywords to search for (all must match)
        field: Which field to search ('title', 'abstract', 'both')
        case_sensitive: Whether search is case-sensitive
        max_results: Maximum number of results to return
        n_workers: Number of parallel workers (default: CPU count)
        
    Returns:
        List of SearchResult objects, sorted by match score
    """
    # Find all page files (recursively)
    page_files = sorted(cache_dir.rglob("page_*.json"))
    
    if not page_files:
        print(f"No page_*.json files found in {cache_dir} (recursive)", file=sys.stderr)
        return []
    
    print(f"Searching {len(page_files)} files for keywords: {keywords}", file=sys.stderr)
    
    all_results = []
    
    # Use ProcessPoolExecutor for parallel search
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all search tasks
        future_to_file = {
            executor.submit(search_file, fp, keywords, field, case_sensitive): fp
            for fp in page_files
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_file):
            completed += 1
            if completed % 500 == 0:
                print(f"  Searched {completed}/{len(page_files)} files...", file=sys.stderr)
            
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                file_path = future_to_file[future]
                print(f"Warning: Error searching {file_path}: {e}", file=sys.stderr)
    
    # Sort by match score (descending), then by title
    all_results.sort(key=lambda r: (-r.match_score, r.display_name))
    
    return all_results[:max_results]


def get_default_cache_dir() -> Path:
    """Get the default cache directory path."""
    # Look relative to this script's location
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    
    # New structure: cache/api_data/17/1702/T*/work_type/page_*.json
    return repo_root / "cache" / "api_data" / "17" / "1702"


def main():
    parser = argparse.ArgumentParser(
        description="Search OpenAlex API cache for works matching keywords.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m artifacts_scripts.curriculum.search_cache "tree of thoughts"
    python -m artifacts_scripts.curriculum.search_cache "attention" "transformer" --field title
    python -m artifacts_scripts.curriculum.search_cache "GPT" --max-results 50
        """,
    )
    parser.add_argument(
        "keywords",
        nargs="+",
        help="Keywords to search for (works must contain ALL keywords)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory containing page_*.json files (default: auto-detect)",
    )
    parser.add_argument(
        "--field",
        choices=["title", "abstract", "both"],
        default="both",
        help="Which field to search (default: both)",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Make search case-sensitive",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of results to return (default: 20)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "ids"],
        default="text",
        help="Output format (default: text)",
    )
    
    args = parser.parse_args()
    
    # Get cache directory
    cache_dir = args.cache_dir or get_default_cache_dir()
    
    if not cache_dir.exists():
        print(f"Error: Cache directory not found: {cache_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Run search
    results = search_cache(
        cache_dir=cache_dir,
        keywords=args.keywords,
        field=args.field,
        case_sensitive=args.case_sensitive,
        max_results=args.max_results,
        n_workers=args.workers,
    )
    
    # Output results
    if args.output_format == "json":
        output = [
            {
                "openalex_id": r.openalex_id,
                "display_name": r.display_name,
                "publication_date": r.publication_date,
                "type": r.work_type,
                "match_score": r.match_score,
            }
            for r in results
        ]
        print(json.dumps(output, indent=2))
    
    elif args.output_format == "ids":
        for r in results:
            print(r.openalex_id)
    
    else:  # text
        print(f"\nFound {len(results)} matching works:\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r.display_name}")
            print(f"   ID: {r.openalex_id}")
            print(f"   Type: {r.work_type}, Date: {r.publication_date}")
            print(f"   Match score: {r.match_score:.2f}")
            if r.abstract_snippet:
                print(f"   Abstract: {r.abstract_snippet}")
            print()


if __name__ == "__main__":
    main()
