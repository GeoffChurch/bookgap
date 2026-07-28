#!/usr/bin/env bash
# Curriculum Showdown P1 — serial pi mini-run batch (agent + hybrid arms).
# One GPU job at a time (gemma4:31b-it-qat); resumable: completed keys are
# skipped via the manifest. Manifest rows: key<TAB>rundir<TAB>exit_code.
set -u
LT="$HOME/src/learn-and-teach"
BRIEFS="$LT/outputs/showdown_p1/briefs"
SHOW="$HOME/src/bookgap/artifacts/gap_detection_eval/showdown"
MANIFEST="$SHOW/p1_runs.tsv"
mkdir -p "$SHOW"
touch "$MANIFEST"

run_one() {  # key brief [extra_stage_csv]
  local key=$1 brief=$2 extra=${3:-}
  if grep -q "^${key}	" "$MANIFEST"; then echo "skip $key (done)"; return 0; fi
  local stage="[apparatus/websearch.py,apparatus/fetch_url.py${extra}]"
  echo "=== $key start $(date +%H:%M:%S)"
  local out rc rundir
  out=$(cd "$LT" && python3 harness/pi_launch.py \
      task.prompt_file="$brief" \
      "task.stage=$stage" \
      pi.model=gemma4:31b-it-qat pi.provider=ollama pi.thinking=high \
      auto_continue.max=3 \
      "auto_continue.until=test -s toc.md" 2>>"$SHOW/p1_minis.stderr.log")
  rc=$?
  rundir=$(printf '%s\n' "$out" | sed -n 's/^RUNDIR=//p' | tail -1)
  printf '%s\t%s\t%s\n' "$key" "$rundir" "$rc" >> "$MANIFEST"
  echo "=== $key done rc=$rc dir=$rundir"
}

for topic in datascience causal postcutoff; do
  for angle in problems methods applications; do
    run_one "mini_${topic}_${angle}" "$BRIEFS/mini_${topic}_${angle}.md"
  done
done
for topic in datascience causal postcutoff; do
  run_one "hybrid_${topic}" "$BRIEFS/hybrid_${topic}.md" ",$SHOW/p1_skeleton_${topic}.md"
done
echo ALL_DONE
