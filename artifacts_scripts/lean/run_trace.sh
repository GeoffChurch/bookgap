#!/usr/bin/env bash
# Launch the LeanDojo mathlib4 trace under bookgap.jobs (systemd-run cgroup) so
# the multi-hour from-source build survives, is tear-down-able as a tree, and is
# monitorable. The trace itself is native (no Docker in lean-dojo 4.x).
#
# Usage:
#   conda run -p .conda python -m bookgap.jobs run --prefix leandojo_trace -- \
#       bash artifacts_scripts/lean/run_trace.sh
#
# The wrapped python's stdout/stderr is captured by both the bookgap.jobs sidecar
# (cache/.jobs/<job_id>.log) and the flushed mirror below.
set -euo pipefail

# elan's lean/lake must be visible: LeanDojo shells out to lake to build mathlib.
export PATH="$HOME/.elan/bin:$PATH"

# Use unbuffered python so the sidecar log shows progress live despite conda run
# buffering. stdbuf belt-and-suspenders for any child processes.
export PYTHONUNBUFFERED=1

REPO=/home/gchurchill/src/bookgap
MIRROR="$REPO/cache/lean/trace.log"
mkdir -p "$(dirname "$MIRROR")"

echo "[$(date -Is)] launching LeanDojo trace; PATH lean=$(command -v lean) lake=$(command -v lake)" | tee -a "$MIRROR"

stdbuf -oL -eL conda run -p "$REPO/.conda" --no-capture-output \
    python -u "$REPO/artifacts_scripts/lean/trace_mathlib.py" 2>&1 | stdbuf -oL tee -a "$MIRROR"
rc="${PIPESTATUS[0]}"

echo "[$(date -Is)] trace_mathlib.py exit=$rc" | tee -a "$MIRROR"
exit "$rc"
