#!/usr/bin/env bash
# Fan out the 3 field-17 panel books in PARALLEL (250G RAM, ~10-12G per cloud, so
# 3 concurrent is safe). Each book = its own run_book + log. CPU-only.
#   bash artifacts_scripts/curriculum/run_close_loop_field17.sh
set -uo pipefail
cd /home/gchurchill/src/bookgap
DRV=artifacts_scripts/curriculum/run_close_loop.sh

declare -A GAP=( [mcs]=mcs_gap [color_image]=color_image_gap [neuromorphic]=neuromorphic_gap )
declare -A TOC=( [mcs]=W4235013513_toc [color_image]=W4211017872_toc [neuromorphic]=W4247813589_toc )
declare -A NCH=( [mcs]=10 [color_image]=24 [neuromorphic]=13 )

pids=()
for slug in mcs color_image neuromorphic; do
  nohup bash "$DRV" run_book "$slug" \
      "artifacts/book_gaps/${GAP[$slug]}.json" \
      "artifacts/book_gaps/${TOC[$slug]}.json" "${NCH[$slug]}" \
      > "/tmp/cl_${slug}.log" 2>&1 &
  pids+=($!)
  echo "launched $slug -> /tmp/cl_${slug}.log (PID $!)"
done
echo "FIELD17_PIDS=${pids[*]}"
