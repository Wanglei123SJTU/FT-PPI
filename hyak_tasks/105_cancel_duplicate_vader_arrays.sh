#!/bin/bash
set -euo pipefail

TASK_ID="105_cancel_duplicate_vader_arrays"
LOCK_DIR=".hyak_runner/task_locks"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/${TASK_ID}.lock"
if ! flock -n 9; then
  echo "${TASK_ID}_already_running_elsewhere"
  exit 0
fi

echo "${TASK_ID}_task_start"
hostname
date
git rev-parse --short HEAD

echo "== CANCEL DUPLICATE VADER ARRAYS =="
echo "Keeping 36219844, which is the active generic ckpt-g2 VADER array from task 104."
for job_id in 36219817 36219881; do
  if squeue -j "$job_id" -h >/dev/null 2>&1 && [ -n "$(squeue -j "$job_id" -h)" ]; then
    echo "cancelling_duplicate_job=$job_id"
    squeue -j "$job_id" || true
    scancel "$job_id" || true
  else
    echo "duplicate_job_not_active=$job_id"
  fi
done

sleep 5
echo "== QUEUE AFTER CANCEL =="
squeue -u "$USER" || true

echo "== VADER METRICS COUNT =="
.venv-hyak/bin/python - <<'PY'
from pathlib import Path
root = Path("/gscratch/scrubbed/lei0603/ft-ppi/artifacts/upworthy_question_vader_current_balanced_focused_1p5b")
print("vader_metrics_count", len(list(root.glob("*/rep_*/s_*/metrics.json"))))
PY

echo "${TASK_ID}_task_done"
