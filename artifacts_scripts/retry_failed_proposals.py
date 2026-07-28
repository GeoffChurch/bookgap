"""One-off: re-run the strict-parse path for proposal-None topics.

Loads the prompt records for the failing topics from disk, calls the
same Ollama strict-parse + one-shot retry path as the main pipeline,
and rewrites the matching JSONL records in-place. A timestamped backup
of the original JSONL is taken first.

Usage::

    uv run python -m artifacts_scripts.retry_failed_proposals \\
        --run-dir outputs/gemma4_sweep/2026-05-22_11-11-43_gist_a10_b5_dropk10 \\
        --topic-ids T10126 T12026
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from bookgap.llm.generators.ollama_local import (
    OllamaProposalGenerator,
    _try_parse_proposal,
)
from bookgap.llm.prompt_record import read_prompt_record
from bookgap.llm.prompt_render import render_prompts


def _retry_one(
    gen: OllamaProposalGenerator, prompt_record_path: Path,
) -> dict:
    """Mirror OllamaProposalGenerator.generate() but skip the
    ProposalContext arg (we don't need it to write the JSONL line).
    """
    record = read_prompt_record(prompt_record_path)
    prompts = render_prompts(record)
    custom_id = record.custom_id

    print(f"[{custom_id}] attempt 1...", flush=True)
    response = gen._chat_once(prompts)
    raw = response.message.content or ""
    thinking = getattr(response.message, "thinking", None)
    proposal = _try_parse_proposal(raw)
    metadata = gen._build_metadata(response)

    prior_attempts: list[dict[str, str | None]] = []
    while proposal is None and len(prior_attempts) < gen.max_retries:
        prior_attempts.append({"raw_content": raw, "thinking": thinking})
        attempt_n = len(prior_attempts) + 1
        print(f"[{custom_id}] strict-parse failed; attempt {attempt_n}...",
              flush=True)
        response = gen._chat_once(prompts)
        raw = response.message.content or ""
        thinking = getattr(response.message, "thinking", None)
        proposal = _try_parse_proposal(raw)
        metadata.update(gen._build_metadata(response))

    if prior_attempts:
        metadata["retried"] = True
        metadata["prior_attempts"] = prior_attempts

    print(
        f"[{custom_id}] result: {'OK' if proposal else 'still None'} "
        f"({len(raw)} raw chars, {len(prior_attempts)} prior attempts)",
        flush=True,
    )

    return {
        "custom_id": custom_id,
        "proposal": proposal.model_dump() if proposal else None,
        "raw_content": raw,
        "thinking": thinking,
        "metadata": metadata,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--topic-ids", nargs="+", required=True)
    ap.add_argument("--gap-idx", type=int, default=0)
    ap.add_argument("--ollama-model", default="gemma4:31b")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--num-predict", type=int, default=8192)
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    jsonl_path = run_dir / "ollama_results.jsonl"
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found", file=sys.stderr)
        return 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = jsonl_path.with_name(f"ollama_results.jsonl.bak.{ts}")
    shutil.copy2(jsonl_path, backup_path)
    print(f"Backed up {jsonl_path} -> {backup_path}", flush=True)

    gen = OllamaProposalGenerator(
        model=args.ollama_model,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )

    fresh_records: dict[str, dict] = {}
    for tid in args.topic_ids:
        prompt_path = run_dir / "prompts" / f"gap_{args.gap_idx}_topic_{tid}.json"
        if not prompt_path.exists():
            print(f"ERROR: prompt file missing for {tid}: {prompt_path}",
                  file=sys.stderr)
            return 2
        record = _retry_one(gen, prompt_path)
        fresh_records[record["custom_id"]] = record

    out_lines: list[str] = []
    replaced = 0
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        cid = d.get("custom_id")
        if cid in fresh_records:
            out_lines.append(json.dumps(fresh_records[cid]))
            replaced += 1
        else:
            out_lines.append(line)
    jsonl_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Rewrote {jsonl_path}: {len(out_lines)} lines, {replaced} replaced",
          flush=True)

    print("Summary:")
    for cid, rec in fresh_records.items():
        status = "OK" if rec["proposal"] else "still None"
        title = (rec["proposal"] or {}).get("title", "")[:80] if rec["proposal"] else ""
        print(f"  {cid}: {status}  {title}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
