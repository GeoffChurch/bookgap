"""Curated PyPI corpus for the Python gap-detection substrate (Stage 1).

Packages chosen for: (1) real internal type/function structure, (2) a real test
suite (the plantable "operation cluster"), and (3) tagged release history for the
temporal arm. Size-stratified (small -> large) so the cheap repos are available
first and the extractor can smoke-test before the big frameworks finish cloning.

Regenerate the clones with:
    python -m artifacts_scripts.gap_recover.adapters.python.fetch_corpus
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]          # .../src/bookgap
REPO_ROOT = ROOT / "cache" / "code_gap" / "python" / "repos"

# (import_name, git_url, n_recent_tags_to_keep_for_temporal). Ordered small -> large.
PACKAGES: list[tuple[str, str, int]] = [
    ("click",        "https://github.com/pallets/click",              8),
    ("typer",        "https://github.com/fastapi/typer",              8),
    ("attrs",        "https://github.com/python-attrs/attrs",         8),
    ("requests",     "https://github.com/psf/requests",               8),
    ("jinja2",       "https://github.com/pallets/jinja",              8),
    ("flask",        "https://github.com/pallets/flask",              8),
    ("rich",         "https://github.com/Textualize/rich",            8),
    ("httpx",        "https://github.com/encode/httpx",               8),
    ("starlette",    "https://github.com/encode/starlette",           8),
    ("fastapi",      "https://github.com/fastapi/fastapi",            8),
    ("tornado",      "https://github.com/tornadoweb/tornado",         8),
    ("black",        "https://github.com/psf/black",                  8),
    ("networkx",     "https://github.com/networkx/networkx",          8),
    ("aiohttp",      "https://github.com/aio-libs/aiohttp",           6),
    ("poetry",       "https://github.com/python-poetry/poetry",       6),
    ("pydantic",     "https://github.com/pydantic/pydantic",          8),
    ("pillow",       "https://github.com/python-pillow/Pillow",       6),
    ("sqlalchemy",   "https://github.com/sqlalchemy/sqlalchemy",      6),
    ("scikit-learn", "https://github.com/scikit-learn/scikit-learn",  6),
    ("scipy",        "https://github.com/scipy/scipy",                4),
    ("numpy",        "https://github.com/numpy/numpy",                4),
    ("sympy",        "https://github.com/sympy/sympy",                4),
    ("matplotlib",   "https://github.com/matplotlib/matplotlib",      4),
    ("pandas",       "https://github.com/pandas-dev/pandas",          4),
]
