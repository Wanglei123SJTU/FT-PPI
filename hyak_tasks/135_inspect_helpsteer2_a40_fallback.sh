#!/bin/bash
set -euo pipefail

echo "inspect_helpsteer2_a40_fallback_task_start $(date)"

cd "${HYAK_RUNNER_REPO_DIR:-$PWD}"
git status --short --branch

JOB_ID="${JOB_ID:-36268601}"
OUTPUT_DIR="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/helpsteer2_lora_worker_quickdiag_a40_fallback"
LOG_PATTERN="/gscratch/scrubbed/${USER}/ft-ppi/logs/hs2-worker-${JOB_ID}_*.out"

echo "job_id=$JOB_ID"
echo "squeue"
squeue -j "$JOB_ID" || true

echo "sacct"
sacct -j "$JOB_ID" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS -P || true

echo "output files"
if [ -d "$OUTPUT_DIR" ]; then
  find "$OUTPUT_DIR" -maxdepth 3 -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %s %p\n" | sort | tail -n 80 || true
  echo "cell_count=$(find "$OUTPUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | wc -l | tr -d ' ')"
else
  echo "output_dir_missing"
fi

echo "slurm logs"
for log in $(ls -1t $LOG_PATTERN 2>/dev/null | head -n 8); do
  echo "---- $log ----"
  tail -n 220 "$log" || true
done

echo "inspect_helpsteer2_a40_fallback_task_done $(date)"
