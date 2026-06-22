#!/bin/bash
set -euo pipefail

JOB_ID="36266204"

echo "cancel_helpsteer2_lora_slow_start $(date)"
echo "job_id=$JOB_ID"

if squeue -j "$JOB_ID" -h | grep -q .; then
  squeue -j "$JOB_ID" || true
  scancel "$JOB_ID" || true
  echo "cancel_requested job_id=$JOB_ID"
else
  echo "job_not_running job_id=$JOB_ID"
fi

sacct -j "$JOB_ID" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS -P || true

echo "cancel_helpsteer2_lora_slow_done $(date)"
