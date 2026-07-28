#!/bin/bash
# Autonomous d-search orchestration loop.
#
# Each iteration:
#   1. Regenerate run_summary CSV from outputs/gemma4_sweep/
#   2. Invoke the planner; if it proposes 0 cells, terminate
#   3. Otherwise launch the proposed batch as a bookgap.jobs run
#   4. Poll until that job exits the `running` state
#   5. Loop back to step 1
#
# Terminates when the planner converges OR after MAX_ITERATIONS rounds.

set -uo pipefail  # -e omitted: we handle errors explicitly so a bad batch
                  #             doesn't kill the whole overnight run

BOOKGAP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_RUN="uv run"

CSV_PATH="$BOOKGAP_DIR/outputs/gemma4_sweep/run_summary_gap0_gap0.csv"
# IMPORTANT: NOT under cache/.jobs/ — that path is globbed by bookgap.jobs
# manifest discovery, which expects every *.json file to be a job manifest dict.
CELLS_JSON="$BOOKGAP_DIR/cache/autonomous_d_search/cells_to_run.json"
LOG_PATH="$BOOKGAP_DIR/cache/autonomous_d_search/loop.log"

MAX_ITERATIONS=20
POLL_INTERVAL_SEC=120

TOPIC_SUBSET="T10028,T10181,T10201,T10639,T10664,T10682,T10711,T10862,T11424,T11512,T11598,T11612,T11901,T12026,T12262,T13062,T13904,T14351,T14381,T14413"

cd "$BOOKGAP_DIR"

# Ensure log directory exists; start fresh log.
mkdir -p "$(dirname "$LOG_PATH")"
{
  echo "===================================================================="
  echo "Autonomous d-search loop started at $(date -Iseconds)"
  echo "PID: $$"
  echo "===================================================================="
} | tee "$LOG_PATH"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$LOG_PATH"
}

regenerate_summary() {
  log "Regenerating run summary CSV..."
  $UV_RUN python -m artifacts_scripts.output_run_summary \
    --outputs-dir outputs/gemma4_sweep \
    --output-file outputs/gemma4_sweep/run_summary_gap0.csv \
    --topic-subset "$TOPIC_SUBSET" \
    --overwrite >> "$LOG_PATH" 2>&1
  return $?
}

run_planner() {
  log "Invoking planner..."
  $UV_RUN python -m artifacts_scripts.autonomous_d_search \
    --csv "$CSV_PATH" \
    --out "$CELLS_JSON" >> "$LOG_PATH" 2>&1
  return $?
}

cells_count() {
  # Counts the number of triples in the JSON file. Defaults to 0 on parse error.
  $UV_RUN python -c "
import json, sys
try:
    print(len(json.load(open('$CELLS_JSON'))))
except Exception:
    print(0)
" 2>/dev/null
}

launch_batch_and_wait() {
  local iter="$1"
  local prefix="d_search_iter_$iter"
  log "Launching batch as bookgap.jobs run (prefix=$prefix)..."

  # Capture stdout so we can extract the assigned job_id.
  local run_output
  run_output=$($UV_RUN python -m bookgap.jobs run --prefix "$prefix" -- \
    $UV_RUN python -m artifacts_scripts.run_cells_from_json "$CELLS_JSON" 2>&1)
  echo "$run_output" >> "$LOG_PATH"

  local job_id
  job_id=$(echo "$run_output" | grep -oE "${prefix}_[0-9]+" | head -1)
  if [[ -z "$job_id" ]]; then
    log "ERROR: failed to extract job_id from launch output; aborting iteration."
    return 1
  fi
  local manifest="$BOOKGAP_DIR/cache/.jobs/$job_id.json"
  log "Launched job_id=$job_id; polling $manifest..."

  # Wait for the manifest to exist (handles a tiny window between
  # `bookgap.jobs run` exit and the manifest being written).
  for ((i=0; i<30; i++)); do
    [[ -f "$manifest" ]] && break
    sleep 2
  done
  if [[ ! -f "$manifest" ]]; then
    log "ERROR: manifest $manifest never appeared; aborting iteration."
    return 1
  fi

  # Poll the manifest's status field directly. Resilient against transient
  # `bookgap.jobs list` failures. Three consecutive non-"running" reads required
  # before declaring done, to absorb the occasional manifest-rewrite glitch.
  local non_running_streak=0
  while [[ $non_running_streak -lt 3 ]]; do
    sleep $POLL_INTERVAL_SEC
    local status
    status=$(python3 -c "import json; print(json.load(open('$manifest')).get('status',''))" 2>/dev/null)
    if [[ "$status" == "running" ]]; then
      non_running_streak=0
    else
      ((non_running_streak++))
      log "$job_id status=$status (streak=$non_running_streak/3)"
    fi
  done
  log "$job_id confirmed not running (final status: $status)."
}

# -- Main loop --------------------------------------------------------------

for ((iter=1; iter<=MAX_ITERATIONS; iter++)); do
  log "===================================================================="
  log "Iteration $iter / $MAX_ITERATIONS"
  log "===================================================================="

  if ! regenerate_summary; then
    log "ERROR: run_summary regeneration failed; aborting loop."
    exit 1
  fi

  if ! run_planner; then
    log "ERROR: planner failed; aborting loop."
    exit 1
  fi

  n=$(cells_count)
  log "Planner proposed $n cells."

  if [[ "$n" -eq 0 ]]; then
    log "Planner converged. Terminating."
    break
  fi

  launch_batch_and_wait "$iter"

  log "Iteration $iter complete; looping."
done

if [[ "$iter" -gt "$MAX_ITERATIONS" ]]; then
  log "Hit MAX_ITERATIONS=$MAX_ITERATIONS without converging. Exiting."
fi

log "===================================================================="
log "Autonomous d-search loop finished at $(date -Iseconds)"
log "===================================================================="
