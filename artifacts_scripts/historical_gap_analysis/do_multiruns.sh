#!/bin/bash

MAX_YEAR=2025
MIN_YEAR=1960

# Generate comma-separated list of years (descending)
YEARS=$(seq $MAX_YEAR -1 $MIN_YEAR | paste -sd, -)

echo "Running analysis for years: $YEARS"

RADIUS_FRACTIONS="0.5 1 2"
echo "Radius fractions: $RADIUS_FRACTIONS"

for p in $RADIUS_FRACTIONS; do
    echo "Starting runs for p=$p"

    # # Standard run
    # LOG_FILE="historical_analysis_${MAX_YEAR}_${MIN_YEAR}_p${p}.log"
    # echo "  Launching standard run -> $LOG_FILE"
    # nohup python -m artifacts_scripts.historical_gap_analysis.historical_gap_analysis --multirun year=$YEARS experiment.gap.radius_fraction=$p > "$LOG_FILE" 2>&1 &

    # CCLIP run
    LOG_FILE_CCLIP="historical_analysis_${MAX_YEAR}_${MIN_YEAR}_p${p}_CCLIP.log"
    echo "  Launching CCLIP run -> $LOG_FILE_CCLIP"
    nohup python -m artifacts_scripts.historical_gap_analysis.historical_gap_analysis --multirun year=$YEARS experiment.gap.radius_fraction=$p experiment.cloud.denoiser=CCLIP experiment.cloud.drop_k=10 > "$LOG_FILE_CCLIP" 2>&1 &
done