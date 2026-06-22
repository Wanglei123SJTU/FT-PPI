#!/bin/bash
set -euo pipefail

JOB_ID="36266204"
LOG_DIR="/gscratch/scrubbed/${USER}/ft-ppi/logs"
OUT_DIR="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/helpsteer2_lora_length_two_feature_pilot_fixed_fast"

echo "inspect_helpsteer2_lora_fast_job_start $(date)"
echo "job_id=$JOB_ID"

echo "squeue"
squeue -j "$JOB_ID" || true

echo "sacct"
sacct -j "$JOB_ID" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,NodeList%24 -P || true

echo "cell json count"
find "$OUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | wc -l | tr -d ' '
echo

echo "recent cell json"
find "$OUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | sort | tail -n 20 || true

echo "slurm log summaries"
for log in $(ls -1 "$LOG_DIR"/hs2-lora-"${JOB_ID}"_*.out 2>/dev/null | sort | head -n 16); do
  echo "---- $log ----"
  grep -E "cell_done|train_lora_cell_done|Traceback|RuntimeError|CUDA|OutOfMemory|Killed|epoch_done|early_stop|tokenizing|max_length|train_size" "$log" || true
  tail -n 100 "$log" || true
done

echo "inspect_helpsteer2_lora_fast_job_done $(date)"
