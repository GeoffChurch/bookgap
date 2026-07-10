#!/bin/bash
# Real-book ToC reconstruction PANEL driver.
#
# Per book:  brute-force gap  ->  [gate on gap size]  ->  curriculum (Track A)
#            ->  Track B materialize (book ToC -> our format).
# Then:      gemma pairwise (Track A vs Track B), keywords-only, 30 phrases.
# Opus judging is dispatched SEPARATELY via the Claude Agent tool on the
# rendered outlines (a bash driver can't call opus); this driver leaves both
# run-dirs ready for that.
#
# Parallel-where-safe:
#  - The citation graph + cloud embeddings are scope-shared and @MEMORY.cache'd
#    (get_cloud_citation_graph: 93s cold at CS-field, 0.1s warm). So we run the
#    FIRST book to completion as a WARMUP (it builds+caches the graph for its
#    scope); only then do we fan out the rest -- otherwise N cold processes each
#    rebuild the graph. NOTE: warmup only covers the first book's scope; if the
#    panel mixes scopes, group same-scope books or expect one cold build/scope.
#  - Curriculum generation is CPU/RAM-bound (~15GB/book at CS-field, ~104GB at
#    domain). MAX_PARALLEL caps concurrency; default 8 (~120GB) for CS-field --
#    set MAX_PARALLEL=2 for domain scope.
#  - PageRank is personalized per book (NOT cacheable) -- the irreducible
#    ~4min/book (CS-field) or ~30min/book (domain) cost.
#  - gemma evals serialize on the single GPU model (cheap, ~1min each).
#
# Usage:   MAX_PARALLEL=8 SEL=--boundary bash artifacts_scripts/curriculum/run_toc_panel.sh
# Launch as a tracked job for long runs:
#   uv run python -m bookgap.jobs run --prefix toc_panel -- bash artifacts_scripts/curriculum/run_toc_panel.sh
set -u
cd /home/gchurchill/src/bookgap
export PYTHONPATH=/home/gchurchill/src/bookgap
PY="uv run python"
LOG=/tmp/toc_panel_master.log
MAX_PARALLEL=${MAX_PARALLEL:-8}     # memory cap; ~15GB/book CS-field, set 2 for domain
SEL=${SEL:---boundary}             # Track A selector (our best); "" for base
MIN_GAP=${MIN_GAP:-30}             # skip books whose own-gap is degenerate (popular-book tiny gap)

# PANEL entries: "book_id|toc_json|slug|scope_flag"  (scope_flag e.g. "--field 17" or "--subfield 1702" or "")
PANEL=(
  "https://openalex.org/W4235013513|artifacts/book_gaps/W4235013513_toc.json|mcs|--field 17"
  "https://openalex.org/W4247813589|artifacts/book_gaps/W4247813589_toc.json|neuromorphic|--field 17"
  "https://openalex.org/W4211017872|artifacts/book_gaps/W4211017872_toc.json|color_image|--field 17"
)

_gap_size() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['book']['gap_size'])" "$1" 2>/dev/null || echo 0; }

gen_book() {  # gap -> gate -> curriculum -> Track B   (CPU/RAM-bound)
  IFS='|' read -r book_id toc slug scope <<< "$1"
  local gap="artifacts/book_gaps/${slug}_gap.json"
  local trackA="${slug}_trackA"
  local trackB="artifacts/curriculum/${slug}_trackB"
  local bl="/tmp/toc_panel_${slug}.log"
  echo "[$(date +%T)] $slug: brute-force gap ($scope)" >> "$LOG"
  $PY -m artifacts_scripts.curriculum.book_own_gap --brute-force --book-id "$book_id" $scope \
      --out "$gap" > "$bl" 2>&1 || { echo "[$(date +%T)] $slug GAP FAILED" >> "$LOG"; return 1; }
  local n; n=$(_gap_size "$gap")
  if [ "${n:-0}" -lt "$MIN_GAP" ]; then
    echo "[$(date +%T)] $slug SKIP: own-gap=$n < $MIN_GAP (degenerate/popular book)" >> "$LOG"; return 2
  fi
  # Match granularity: our curriculum gets the SAME number of chapters as the book's ToC.
  local ntopics; ntopics=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))['chapters']))" "$toc" 2>/dev/null || echo 15)
  echo "[$(date +%T)] $slug: gap=$n -> curriculum $SEL (desired_n_topics=$ntopics, matched to ToC)" >> "$LOG"
  $PY -m artifacts_scripts.curriculum.book_curriculum --gap-json "$gap" $SEL \
      --desired-n-topics "$ntopics" --output-dir "$trackA" >> "$bl" 2>&1 \
      || { echo "[$(date +%T)] $slug CURRICULUM FAILED" >> "$LOG"; return 1; }
  echo "[$(date +%T)] $slug: Track B materialize (mean-center, phrase-mined)" >> "$LOG"
  $PY -m artifacts_scripts.curriculum.book_toc_reconstruction --gap-json "$gap" --toc-json "$toc" \
      --curriculum-dir "artifacts/curriculum/$trackA" --mean-center --materialize-dir "$trackB" \
      --out "artifacts/book_gaps/${slug}_recon.json" >> "$bl" 2>&1 \
      || { echo "[$(date +%T)] $slug TRACKB FAILED" >> "$LOG"; return 1; }
  echo "[$(date +%T)] $slug: GEN DONE" >> "$LOG"
}

eval_book() {  # gemma pairwise   (GPU, serial)
  IFS='|' read -r book_id toc slug scope <<< "$1"
  [ -f "artifacts/curriculum/${slug}_trackB/summary.md" ] || { echo "[$(date +%T)] $slug: no Track B, skip eval" >> "$LOG"; return; }
  $PY -m artifacts_scripts.curriculum.eval_pairwise_judge \
      --run-dir-a "artifacts/curriculum/${slug}_trackA" --run-dir-b "artifacts/curriculum/${slug}_trackB" \
      --label-a ours --label-b book_toc --model gemma4:31b --repeats 3 --n-titles 0 \
      --out "artifacts/curriculum_eval_pairwise/${slug}_pairwise.json" >> "/tmp/toc_panel_${slug}.log" 2>&1
  echo "[$(date +%T)] $slug: GEMMA DONE -> ${slug}_pairwise.json" >> "$LOG"
}

{
echo "[$(date)] PANEL start: ${#PANEL[@]} books | max_parallel=$MAX_PARALLEL | sel='${SEL}' | min_gap=$MIN_GAP"
# WARMUP: first book to completion (builds + caches citation graph for its scope)
gen_book "${PANEL[0]}"
# fan out the rest, memory-capped
for entry in "${PANEL[@]:1}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 15; done
  gen_book "$entry" &
done
wait
echo "[$(date)] all generation done; gemma evals (serial on GPU)..."
for entry in "${PANEL[@]}"; do eval_book "$entry"; done
echo "[$(date)] PANEL complete. Track A/B run-dirs ready; render + dispatch opus next."
} >> "$LOG" 2>&1
