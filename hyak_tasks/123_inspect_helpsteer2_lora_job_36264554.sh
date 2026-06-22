#!/bin/bash
set -euo pipefail

JOB_ID="36264554"
LOG_DIR="/gscratch/scrubbed/${USER}/ft-ppi/logs"
OUT_DIR="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/helpsteer2_lora_length_two_feature_pilot"

echo "inspect_helpsteer2_lora_job_start $(date)"
echo "job_id=$JOB_ID"

echo "squeue"
squeue -j "$JOB_ID" || true

echo "sacct"
sacct -j "$JOB_ID" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS -P || true

echo "completed cells"
find "$OUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | wc -l | tr -d ' '
echo

echo "slurm log tails"
for log in $(ls -1 "$LOG_DIR"/hs2-lora-"${JOB_ID}"_*.out 2>/dev/null | head -n 12); do
  echo "---- $log ----"
  tail -n 160 "$log" || true
done

echo "inspect_helpsteer2_lora_job_done $(date)"
