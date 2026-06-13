#!/bin/bash
set -euo pipefail

echo "mse_batch_probe_task_start"
hostname
date
git rev-parse --short HEAD

mkdir -p logs

echo "== GPU STATUS =="
sinfo -o "%20P %18G %8D %8t %10C %10m %N" | grep -Ei 'gpu|ckpt|h200|a100|a40|l40|rtx6k' || true
squeue -u "$USER" || true

echo "== SUBMIT MSE BATCH PROBE =="
submit_output=""
submit_status=1
for args in \
  "--partition=ckpt-g2 --gres=gpu:h200:1" \
  "--partition=gpu-h200 --gres=gpu:h200:1" \
  "--partition=ckpt-all --gres=gpu:h200:1" \
  "--partition=ckpt --gres=gpu:a40:1" \
  "--partition=ckpt --gres=gpu:l40s:1" \
  "--partition=ckpt --gres=gpu:rtx6k:1" \
  "--partition=ckpt --gres=gpu:1"
do
  echo "Trying: sbatch $args slurm/run_mse_batch_probe.sbatch"
  set +e
  submit_output=$(sbatch $args slurm/run_mse_batch_probe.sbatch 2>&1)
  submit_status=$?
  set -e
  echo "$submit_output"
  if [ "$submit_status" -eq 0 ]; then
    break
  fi
done

if [ "$submit_status" -ne 0 ]; then
  echo "All MSE batch probe submit attempts failed."
  exit 1
fi

JOB_ID=$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/ {print $4}' | tail -1)
if [ -z "$JOB_ID" ]; then
  echo "Could not parse job id from sbatch output."
  exit 1
fi
echo "JOB_ID=$JOB_ID"
JOB_LOG="logs/wine-msebatch-${JOB_ID}.out"

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

root = Path("artifacts/mse_batch_probe")
summary = root / "summary"
required = [
    summary / "allocation_curve.csv",
    summary / "prediction_diagnostics.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))

curve = pd.read_csv(summary / "allocation_curve.csv")
diagnostics = pd.read_csv(summary / "prediction_diagnostics.csv")
expected = {"batch2_accum4", "batch4_accum2", "batch8_accum1", "batch16_accum1"}
ppi_plus = curve[curve["method"].astype(str).str.contains("ppi_plus", regex=False)].copy()
if set(ppi_plus["variant"].astype(str)) != expected:
    raise SystemExit("missing batch variants")
if not np.isfinite(ppi_plus["normalized_estimated_variance"]).all():
    raise SystemExit("non-finite normalized variance")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)
print("mse batch ppi_plus")
print(
    ppi_plus[[
        "variant",
        "loss",
        "train_size",
        "mean_residual_variance",
        "mean_estimated_variance",
        "normalized_estimated_variance",
        "mean_sample_savings",
        "mean_ci_length",
    ]].sort_values(["normalized_estimated_variance", "variant"]).to_string(index=False)
)

target_diag = diagnostics[diagnostics["role"].isin(["population", "correction"])].sort_values(["role", "variant"])
print("mse batch prediction diagnostics")
print(
    target_diag[[
        "variant",
        "loss",
        "role",
        "mean_corr",
        "mean_rmse",
        "mean_bias",
        "mean_pred_mean",
        "mean_pred_std",
        "mean_residual_variance",
    ]].to_string(index=False)
)
PY

echo "mse_batch_probe_task_done"
