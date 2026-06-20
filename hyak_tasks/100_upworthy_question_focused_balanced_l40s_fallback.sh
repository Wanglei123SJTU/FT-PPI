#!/bin/bash
set -euo pipefail

TASK_ID="100_upworthy_question_focused_balanced_l40s_fallback"
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

OLD_JOB_ID="${OLD_JOB_ID:-36219465}"
if squeue -j "$OLD_JOB_ID" -h >/dev/null 2>&1 && [ -n "$(squeue -j "$OLD_JOB_ID" -h)" ]; then
  echo "cancelling_pending_or_running_job=$OLD_JOB_ID"
  squeue -j "$OLD_JOB_ID" || true
  scancel "$OLD_JOB_ID" || true
  sleep 10
fi

export ARRAY_CONC="${ARRAY_CONC:-8}"
export HYAK_FORCE_GPU_ARGS="${HYAK_FORCE_GPU_ARGS:---partition=ckpt-g2 --gres=gpu:l40s:1}"

echo "forced_gpu_args=$HYAK_FORCE_GPU_ARGS"
echo "== CHECK FOCUSED UPWORTHY STATIC DATA COLUMNS =="
.venv-hyak/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("Data/upworthy_pairs_with_text_features.csv")
required = {
    "delta_QUESTION",
    "delta_NUMERIC",
    "delta_SIMPLICITY",
    "delta_COMMON",
    "delta_LENGTH",
    "delta_VADER_COMPOUND",
    "clicks_a",
    "clicks_b",
    "impressions_a",
    "impressions_b",
}
if not path.exists():
    raise SystemExit("missing Data/upworthy_pairs_with_text_features.csv")
cols = set(pd.read_csv(path, nrows=1).columns)
missing = sorted(required - cols)
if missing:
    raise SystemExit("missing required columns: " + ", ".join(missing))
print("data_ok columns_present=" + ",".join(sorted(required)))
PY

CONFIGS=(
  "configs/upworthy_question_full5_current_balanced_focused_1p5b.yaml"
  "configs/upworthy_question_vader_current_balanced_focused_1p5b.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "== RUN L40S FALLBACK FOCUSED QUESTION PILOT: $cfg =="
  export CONFIG="$cfg"
  bash hyak_tasks/080_upworthy_question_scaling_diagnostic_full.sh
done

echo "${TASK_ID}_task_done"
