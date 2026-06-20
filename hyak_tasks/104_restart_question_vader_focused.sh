#!/bin/bash
set -euo pipefail

TASK_ID="104_restart_question_vader_focused"
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

CONFIG="configs/upworthy_question_vader_current_balanced_focused_1p5b.yaml"
ROOT="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/upworthy_question_vader_current_balanced_focused_1p5b"

echo "== CANCEL STUCK FOCUSED QUESTION ARRAYS =="
for job_id in 36219548 36219549; do
  if squeue -j "$job_id" -h >/dev/null 2>&1 && [ -n "$(squeue -j "$job_id" -h)" ]; then
    echo "cancelling_job=$job_id"
    squeue -j "$job_id" || true
    scancel "$job_id" || true
  else
    echo "job_not_active=$job_id"
  fi
done
sleep 5
squeue -u "$USER" || true

echo "== CURRENT VADER METRICS COUNT =="
.venv-hyak/bin/python - <<'PY'
from pathlib import Path
root = Path("/gscratch/scrubbed/lei0603/ft-ppi/artifacts/upworthy_question_vader_current_balanced_focused_1p5b")
print("vader_metrics_count", len(list(root.glob("*/rep_*/s_*/metrics.json"))))
PY

echo "== RUN QUESTION + VADER FOCUSED PILOT =="
export CONFIG
export ARRAY_CONC="${ARRAY_CONC:-8}"
export HYAK_FORCE_GPU_ARGS="${HYAK_FORCE_GPU_ARGS:---partition=ckpt-g2 --gres=gpu:1}"
bash hyak_tasks/080_upworthy_question_scaling_diagnostic_full.sh

echo "== REFRESH PARTIAL METRICS AFTER VADER RUN =="
bash hyak_tasks/101_upworthy_question_partial_metrics.sh

echo "${TASK_ID}_task_done"
