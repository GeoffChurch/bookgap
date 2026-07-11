"""Regression test for the parents[N] repo-root arithmetic broken by the T9
relocation (gap_recover moved up one level; several ``ROOT``/``bookgap_root``
computations weren't adjusted and silently resolved one level too high).

Dependency-free by design: rather than importing the adapter modules
(some pull in optional rust/bazel/go/lean toolchain plumbing at import time),
this recomputes each file's own ``Path(__file__).resolve().parents[N])``
expression directly from that file's on-disk path, and checks it against the
repo root computed independently from *this* test file's own location.
"""
from pathlib import Path

# tests/gap_recover/test_paths.py -> parents[0]=tests/gap_recover,
# parents[1]=tests, parents[2]=repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# adapters/<lang>/file.py -> parents[0]=adapters/<lang>, [1]=adapters,
# [2]=gap_recover, [3]=artifacts_scripts, [4]=repo root.
_ADAPTER_ROOT_FILES = [
    "artifacts_scripts/gap_recover/adapters/python/corpus.py",
    "artifacts_scripts/gap_recover/adapters/rust/corpus.py",
    "artifacts_scripts/gap_recover/adapters/bazel/corpus.py",
    "artifacts_scripts/gap_recover/adapters/python/build_births.py",
    "artifacts_scripts/gap_recover/adapters/rust/build_births.py",
    "artifacts_scripts/gap_recover/adapters/bazel/build_births.py",
    "artifacts_scripts/gap_recover/adapters/go/build_births.py",
    "artifacts_scripts/gap_recover/adapters/python/build_data.py",
    "artifacts_scripts/gap_recover/adapters/rust/build_data.py",
    "artifacts_scripts/gap_recover/adapters/lean/build_data.py",
]

# coverage_oracle_multi.py -> parents[0]=gap_recover, [1]=artifacts_scripts,
# [2]=repo root.
_COVERAGE_ORACLE_FILE = "artifacts_scripts/gap_recover/coverage_oracle_multi.py"


def test_adapter_root_files_resolve_to_repo_root():
    for rel in _ADAPTER_ROOT_FILES:
        f = (REPO_ROOT / rel).resolve()
        assert f.is_file(), f"expected file at {f}"
        computed_root = f.parents[4]
        assert computed_root == REPO_ROOT, (
            f"{rel}: parents[4] = {computed_root}, expected repo root {REPO_ROOT}"
        )


def test_coverage_oracle_multi_bookgap_root_resolves_to_repo_root():
    f = (REPO_ROOT / _COVERAGE_ORACLE_FILE).resolve()
    assert f.is_file(), f"expected file at {f}"
    computed_root = f.parents[2]
    assert computed_root == REPO_ROOT, (
        f"{_COVERAGE_ORACLE_FILE}: parents[2] = {computed_root}, "
        f"expected repo root {REPO_ROOT}"
    )
