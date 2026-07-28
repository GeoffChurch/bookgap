#!/usr/bin/env bash
# Curriculum Showdown P1 — generic serial pi job runner (picks / judges).
# usage: p1_run_jobs.sh <jobs.tsv> <manifest.tsv> [auto_continue_max]
# jobs.tsv rows: key<TAB>brief_abs_path<TAB>until_predicate<TAB>stage_csv_or_-
# Judges/picks run OFFLINE (no web tools staged; pi.offline=true).
set -u
JOBS=$(readlink -f "$1")
MANIFEST=$(readlink -f "$2")
ACMAX=${3:-2}
LT="$HOME/src/learn-and-teach"
mkdir -p "$(dirname "$MANIFEST")"
touch "$MANIFEST"

while IFS=$'\t' read -r key brief until stage; do
  [ -z "${key:-}" ] && continue
  if grep -q "^${key}	" "$MANIFEST"; then echo "skip $key (done)"; continue; fi
  args=(task.prompt_file="$brief" pi.model=gemma4:31b-it-qat pi.provider=ollama
        pi.thinking=high pi.offline=true auto_continue.max="$ACMAX"
        "auto_continue.until=$until")
  if [ "$stage" != "-" ]; then
    args+=("task.stage=[$stage]")
  fi
  echo "=== $key start $(date +%H:%M:%S)"
  # </dev/null: pi must NOT inherit the loop's stdin (it would drain the jobs file)
  out=$(cd "$LT" && python3 harness/pi_launch.py "${args[@]}" \
        2>>"${MANIFEST%.tsv}.stderr.log" </dev/null)
  rc=$?
  rundir=$(printf '%s\n' "$out" | sed -n 's/^RUNDIR=//p' | tail -1)
  printf '%s\t%s\t%s\n' "$key" "$rundir" "$rc" >> "$MANIFEST"
  echo "=== $key done rc=$rc dir=$rundir"
done < "$JOBS"
echo ALL_DONE
