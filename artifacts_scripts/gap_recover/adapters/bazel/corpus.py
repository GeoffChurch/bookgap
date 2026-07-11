"""Bazel monorepo corpus — the CLEAN + COARSE substrate (the granularity cross).

Nodes = build targets; B = ``cc_library`` ("concepts"), F = ``cc_test``/``cc_binary``
("operations"); edges = build ``deps``. Build graphs are EXACT (no static-resolution
noise) but COARSE (target, not function), so this stage crosses cleanness with
granularity. Extracted via ``bazel query --output=xml`` (parses BUILD files; no compile).

    python -m artifacts_scripts.gap_recover.adapters.bazel.build_data --out cache/code_gap/bazel/data
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = ROOT / "cache" / "code_gap" / "bazel"
BAZELISK = "/home/gchurchill/go/bin/bazelisk"

# (name, git_url, query_scope)
PACKAGES: list[tuple[str, str, str]] = [
    ("abseil-cpp", "https://github.com/abseil/abseil-cpp",        "//absl/..."),
    ("protobuf",   "https://github.com/protocolbuffers/protobuf", "//src/..."),
    ("grpc",       "https://github.com/grpc/grpc",                "//:all"),
]
