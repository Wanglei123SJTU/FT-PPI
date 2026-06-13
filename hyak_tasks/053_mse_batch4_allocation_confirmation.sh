#!/bin/bash
set -euo pipefail

echo "mse_batch4_allocation_confirmation_task_start"
hostname
date
git rev-parse --short HEAD

mkdir -p logs

echo "== GPU STATUS =="
sinfo -o "%20P %18G %8D %8t %10C %10m %N" | grep -Ei 'gpu|ckpt|h200|a100|a40|l40|rtx6k' || true
squeue -u "$USER" || true

echo "== SUBMIT MSE BATCH4 ALLOCATION CONFIRMATION =="
submit_output=""
submit_status=1
JOB_ID=""
QUEUE_CHECK_SECONDS="${HYAK_QUEUE_CHECK_SECONDS:-600}"
QUEUE_CHECK_POLL_SECONDS="${HYAK_QUEUE_CHECK_POLL_SECONDS:-60}"
for args in \
  "--partition=ckpt-g2 --gres=gpu:h200:1" \
  "--partition=gpu-h200 --gres=gpu:h200:1" \
  "--partition=ckpt-all --gres=gpu:h200:1" \
  "--partition=ckpt --gres=gpu:a40:1" \
  "--partition=ckpt --gres=gpu:l40s:1" \
  "--partition=ckpt --gres=gpu:rtx6k:1" \
  "--partition=ckpt --gres=gpu:1"
do
  echo "Trying: sbatch $args slurm/run_mse_batch4_allocation_confirmation.sbatch"
  set +e
  submit_output=$(sbatch $args slurm/run_mse_batch4_allocation_confirmation.sbatch 2>&1)
  submit_status=$?
  set -e
  echo "$submit_output"
  if [ "$submit_status" -eq 0 ]; then
    candidate_job_id=$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/ {print $4}' | tail -1)
    if [ -z "$candidate_job_id" ]; then
      echo "Could not parse candidate job id from sbatch output."
      submit_status=1
      continue
    fi

    echo "CANDIDATE_JOB_ID=$candidate_job_id"
    queue_start=$(date +%s)
    while true; do
      queue_line=$(squeue -j "$candidate_job_id" -h -o "%T|%R" 2>/dev/null | head -1 || true)
      if [ -z "$queue_line" ]; then
        echo "Candidate job left queue during initial check; accepting it."
        JOB_ID="$candidate_job_id"
        break
      fi
      state=${queue_line%%|*}
      reason=${queue_line#*|}
      echo "candidate_state=$state reason=$reason"
      if [ "$state" != "PENDING" ]; then
        JOB_ID="$candidate_job_id"
        break
      fi
      queue_elapsed=$(( $(date +%s) - queue_start ))
      if [ "$queue_elapsed" -ge "$QUEUE_CHECK_SECONDS" ]; then
        echo "Candidate still pending after ${QUEUE_CHECK_SECONDS}s; canceling and trying next option."
        scancel "$candidate_job_id" || true
        submit_status=1
        break
      fi
      sleep "$QUEUE_CHECK_POLL_SECONDS"
    done

    if [ -n "$JOB_ID" ]; then
      break
    fi
  fi
done

if [ "$submit_status" -ne 0 ]; then
  echo "All MSE batch4 allocation submit attempts failed."
  exit 1
fi

if [ -z "$JOB_ID" ]; then
  echo "Could not parse job id from sbatch output."
  exit 1
fi
echo "JOB_ID=$JOB_ID"
JOB_LOG="logs/wine-b4alloc-${JOB_ID}.out"

echo "== MONITOR =="
while squeue -j "$JOB_ID" -h >/dev/null 2>&1 && [ -n "$(squeue -j "$JOB_ID" -h)" ]; do
  date
  squeue -j "$JOB_ID" || true
  if [ -f "$JOB_LOG" ]; then
    echo "--- tail $JOB_LOG ---"
    tail -180 "$JOB_LOG" || true
  else
    echo "$JOB_LOG not created yet"
  fi
  sleep 180
done

echo "== FINAL STATUS =="
sacct -j "$JOB_ID" --format=JobID,JobName%25,Partition,State,ExitCode,Elapsed,MaxRSS -P 2>/dev/null || true
if [ -f "$JOB_LOG" ]; then
  echo "--- final tail $JOB_LOG ---"
  tail -360 "$JOB_LOG"
else
  echo "$JOB_LOG missing"
  exit 1
fi

echo "== VALIDATE OUTPUTS =="
.venv-hyak/bin/python - <<'PY'
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.allocation_regret import summarize_rampup_allocation_regret

root = Path("artifacts/mse_batch4_allocation_confirmation")
summary = root / "summary"
required = [
    summary / "allocation_metrics.csv",
    summary / "allocation_curve.csv",
    summary / "oracle_allocations.csv",
    summary / "prediction_diagnostics.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))

metrics = pd.read_csv(summary / "allocation_metrics.csv")
curve = pd.read_csv(summary / "allocation_curve.csv")
diagnostics = pd.read_csv(summary / "prediction_diagnostics.csv")
if metrics.empty or curve.empty or diagnostics.empty:
    raise SystemExit("summary tables are empty")
expected_train_sizes = {25, 50, 75, 100}
ppi_plus = curve[curve["method"].astype(str).str.contains("ppi_plus", regex=False)].copy()
if set(ppi_plus["train_size"].astype(int)) != expected_train_sizes:
    raise SystemExit("missing ppi_plus train sizes")
if not np.isfinite(ppi_plus["normalized_estimated_variance"]).all():
    raise SystemExit("non-finite normalized variance")

regret = summarize_rampup_allocation_regret(curve)
regret.to_csv(summary / "rampup_allocation_regret.csv", index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)
print("mse batch4 allocation ppi_plus")
print(
    ppi_plus[[
        "budget",
        "allocation_ratio",
        "train_size",
        "validation_size",
        "n_replications",
        "mean_residual_variance",
        "mean_estimated_variance",
        "normalized_estimated_variance",
        "mean_sample_savings",
        "mean_ci_length",
    ]].sort_values("train_size").to_string(index=False)
)
print("mse batch4 rampup allocation regret")
print(regret.to_string(index=False))

target_diag = diagnostics[(diagnostics["role"].isin(["population", "correction"])) & (diagnostics["train_size"].isin(expected_train_sizes))]
print("mse batch4 prediction diagnostics")
print(
    target_diag[[
        "loss",
        "train_size",
        "role",
        "mean_corr",
        "mean_rmse",
        "mean_bias",
        "mean_pred_mean",
        "mean_pred_std",
        "mean_residual_variance",
    ]].sort_values(["train_size", "role"]).to_string(index=False)
)
PY

echo "mse_batch4_allocation_confirmation_task_done"
