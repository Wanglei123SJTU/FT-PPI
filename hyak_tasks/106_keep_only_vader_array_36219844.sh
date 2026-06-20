#!/bin/bash
set -euo pipefail

TASK_ID="106_keep_only_vader_array_36219844"
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

KEEP_JOB="36219844"

echo "== ACTIVE UPWORTHY JOBS BEFORE CLEANUP =="
squeue -u "$USER" -n upworthy || true

echo "== CANCEL ALL UPWORTHY SLURM PARENTS EXCEPT ${KEEP_JOB} =="
squeue -u "$USER" -n upworthy -h -o "%A" 2>/dev/null | sort -u | while read -r job_id; do
  [ -n "$job_id" ] || continue
  if [ "$job_id" = "$KEEP_JOB" ]; then
    echo "keeping_job=$job_id"
  else
    echo "cancelling_extra_upworthy_job=$job_id"
    scancel "$job_id" || true
  fi
done

sleep 5
echo "== ACTIVE UPWORTHY JOBS AFTER CLEANUP =="
squeue -u "$USER" -n upworthy || true

echo "== VADER METRICS COUNT =="
.venv-hyak/bin/python - <<'PY'
from pathlib import Path
root = Path("/gscratch/scrubbed/lei0603/ft-ppi/artifacts/upworthy_question_vader_current_balanced_focused_1p5b")
print("vader_metrics_count", len(list(root.glob("*/rep_*/s_*/metrics.json"))))
PY

echo "${TASK_ID}_task_done"
