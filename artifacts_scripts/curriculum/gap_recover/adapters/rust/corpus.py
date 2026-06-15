"""Curated Rust corpus for the 4th-language modality check (Stage 3).

Mid-size, mostly-regular-Rust library crates (not all proc-macro), each with types,
functions/methods, traits, and tests — parsed from source with tree-sitter-rust (NO
cargo build / no Rust toolchain needed). Size-stratified small -> large.

    python -m artifacts_scripts.curriculum.gap_recover.adapters.rust.fetch_corpus
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]            # .../src/bookgap
REPO_ROOT = ROOT / "cache" / "code_gap" / "rust" / "repos"

# (name, git_url, n_recent_tags). Ordered small -> large.
PACKAGES: list[tuple[str, str, int]] = [
    ("bitflags",   "https://github.com/bitflags/bitflags",       6),
    ("anyhow",     "https://github.com/dtolnay/anyhow",          6),
    ("semver",     "https://github.com/dtolnay/semver",          6),
    ("smallvec",   "https://github.com/servo/rust-smallvec",     6),
    ("itertools",  "https://github.com/rust-itertools/itertools",6),
    ("bytes",      "https://github.com/tokio-rs/bytes",          6),
    ("log",        "https://github.com/rust-lang/log",           6),
    ("url",        "https://github.com/servo/rust-url",          6),
    ("indexmap",   "https://github.com/indexmap-rs/indexmap",    6),
    ("toml",       "https://github.com/toml-rs/toml",            4),
    ("regex",      "https://github.com/rust-lang/regex",         6),
    ("rayon",      "https://github.com/rayon-rs/rayon",          6),
    ("clap",       "https://github.com/clap-rs/clap",            6),
    ("hyper",      "https://github.com/hyperium/hyper",          6),
    ("chrono",     "https://github.com/chronotope/chrono",       6),
    ("ripgrep",    "https://github.com/BurntSushi/ripgrep",      4),
    ("serde_json", "https://github.com/serde-rs/json",           4),
    ("tokio",      "https://github.com/tokio-rs/tokio",          4),
]
