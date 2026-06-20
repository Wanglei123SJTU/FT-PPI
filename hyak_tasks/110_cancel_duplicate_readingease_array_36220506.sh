#!/bin/bash
set -euo pipefail

TASK_ID="110_cancel_duplicate_readingease_array_36220506"

echo "${TASK_ID}_task_start"
hostname
date
git rev-parse --short HEAD

echo "== QUEUE BEFORE DUPLICATE CANCEL =="
squeue -u "$USER" || true

if squeue -j 36220506 -h >/dev/null 2>&1 && [ -n "$(squeue -j 36220506 -h)" ]; then
  echo "cancelling_duplicate_job=36220506"
  scancel 36220506 || true
else
  echo "duplicate_job_36220506_not_present"
fi

echo "== QUEUE AFTER DUPLICATE CANCEL =="
squeue -u "$USER" || true

echo "${TASK_ID}_task_done"
