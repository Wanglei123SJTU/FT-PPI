#!/bin/bash
set -euo pipefail

echo "114_cancel_duplicate_full5_array_task_start"
hostname
date
git rev-parse --short HEAD

KEEP_JOB="36222501"
CANCEL_JOB="36222500"

echo "== QUEUE BEFORE CANCEL =="
squeue -u "$USER" | grep -E "${CANCEL_JOB}|${KEEP_JOB}" || true

if squeue -j "$CANCEL_JOB" -h >/dev/null 2>&1 && [ -n "$(squeue -j "$CANCEL_JOB" -h)" ]; then
  echo "canceling_duplicate_job=${CANCEL_JOB}"
  scancel "$CANCEL_JOB" || true
else
  echo "duplicate_job_not_present=${CANCEL_JOB}"
fi

sleep 5

echo "== QUEUE AFTER CANCEL =="
squeue -u "$USER" | grep -E "${CANCEL_JOB}|${KEEP_JOB}" || true

echo "114_cancel_duplicate_full5_array_task_done"
