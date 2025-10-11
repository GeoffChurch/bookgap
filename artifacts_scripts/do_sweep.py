from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

LOGGER = logging.getLogger(__name__)


@dataclass
class SweepPatch:
    overrides: Dict[str, Any] = field(default_factory=dict)

    def override_strings(self) -> List[str]:
        items = list(self.overrides.items())
        return [format_override(key, value) for key, value in items]


DEFAULT_PATCH_FILE = Path(__file__).resolve().parent / "do_sweep_patches.jsonl"


def format_override(key: str, value: Any) -> str:
    if isinstance(value, Mapping) and "_raw" in value:
        raw_value = value["_raw"]
        if not isinstance(raw_value, str):
            raise TypeError(f"_raw value for {key!r} must be a string")
        return f"{key}={raw_value}"

    if isinstance(value, str):
        return f"{key}={json.dumps(value)}"
    if isinstance(value, bool):
        return f"{key}={'true' if value else 'false'}"
    if value is None:
        return f"{key}=null"

    return f"{key}={value}"


def parse_patch_mapping(mapping: Mapping[str, Any]) -> SweepPatch:
    return SweepPatch(overrides=dict(mapping))


def load_patches_from_json_text(text: str) -> List[SweepPatch]:
    data = json.loads(text)

    if isinstance(data, list):
        patches: List[SweepPatch] = []
        for item in data:
            if not isinstance(item, Mapping):
                raise argparse.ArgumentTypeError(
                    "Patch list entries must be JSON objects with override key/value pairs"
                )
            patches.append(parse_patch_mapping(item))
        return patches

    if isinstance(data, Mapping):
        return [parse_patch_mapping(data)]

    raise argparse.ArgumentTypeError(
        "Patch specification must be a JSON object or list of objects"
    )


def load_patches_from_jsonl(path: Path) -> List[SweepPatch]:
    patches: List[SweepPatch] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:  # pragma: no cover - IO failure
        raise argparse.ArgumentTypeError(f"Failed to read patch file {path}: {exc}") from exc

    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid JSON on line {index} in {path}: {exc}"
            ) from exc
        if not isinstance(data, Mapping):
            raise argparse.ArgumentTypeError(
                f"Line {index} in {path} must be a JSON object with override mappings"
            )
        patches.append(parse_patch_mapping(data))

    return patches


def parse_patch_file(path: Path) -> List[SweepPatch]:
    if path.suffix == ".jsonl":
        return load_patches_from_jsonl(path)

    # try:
    #     text = path.read_text(encoding="utf-8")
    # except OSError as exc:  # pragma: no cover - IO failure
    #     raise argparse.ArgumentTypeError(f"Failed to read patch file {path}: {exc}") from exc
    # return load_patches_from_json_text(text)


def build_patch_plan(args: argparse.Namespace) -> List[SweepPatch]:
    patch_path: Path | None = args.patch_file

    if patch_path is None:
        patch_path = DEFAULT_PATCH_FILE

    if not patch_path.exists():
        LOGGER.error("Patch file %s not found.", patch_path)
        raise SystemExit(1)

    try:
        patches = parse_patch_file(patch_path)
    except argparse.ArgumentTypeError as exc:
        LOGGER.error("Failed to load patch file %s: %s", patch_path, exc)
        raise SystemExit(1) from exc

    if not patches:
        LOGGER.warning("Patch file %s did not produce any overrides.", patch_path)

    return patches


def format_command(command: Sequence[str]) -> str:
    return shlex.join(command)


def run_sweep(
    patches: Iterable[SweepPatch],
    config_name: str | None,
    base_overrides: Sequence[str],
    dry_run: bool,
    stop_on_error: bool,
) -> None:
    base_command: List[str] = [sys.executable, "-m", "bookgap.batch_generator"]
    if config_name:
        base_command.extend(["--config-name", config_name])

    patch_list = list(patches)
    total = len(patch_list)

    for index, patch in enumerate(patch_list, start=1):
        command = list(base_command)
        command.extend(base_overrides)
        patch_override_strings = patch.override_strings()
        command.extend(patch_override_strings)

        message = (
            f"Sweep {index}/{total}: "
            + ", ".join(patch_override_strings)
        )

        if dry_run:
            LOGGER.info("[DRY RUN] %s => %s", message, format_command(command))
            continue

        LOGGER.info("%s", message)
        result = subprocess.run(command, check=False)

        if result.returncode != 0:
            LOGGER.error(
                "Run %d failed with exit code %d: %s",
                index,
                result.returncode,
                format_command(command),
            )
            if stop_on_error:
                raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a sequence of BookGap batch_generator jobs with configurable override patches."
        )
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Optional Hydra config name to use for each run (e.g., anti_csp_drop1).",
    )
    parser.add_argument(
        "--override",
        dest="overrides",
        action="append",
        default=None,
        help=(
            "Additional Hydra override expressions applied to every run before patch-specific overrides. "
            "Example: --override gap.radius_fraction=0.8"
        ),
    )
    parser.add_argument(
        "--patch-file",
        dest="patch_file",
        type=Path,
        default=None,
        help=(
            "Path to a JSONL file containing one JSON object per line with override mappings."
            f" Defaults to {DEFAULT_PATCH_FILE}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed without running them.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the sweep immediately if any run exits with a non-zero status.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    base_overrides: List[str] = args.overrides or []
    patches = build_patch_plan(args)

    if not patches:
        LOGGER.warning("No sweep patches specified; nothing to do.")
        return

    run_sweep(
        patches=patches,
        config_name=args.config_name,
        base_overrides=base_overrides,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
    )


if __name__ == "__main__":
    main()
