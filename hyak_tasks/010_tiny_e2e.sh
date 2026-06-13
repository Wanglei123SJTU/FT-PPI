#!/bin/bash
set -euo pipefail

echo "tiny_e2e_task_start"
hostname
date
git rev-parse --short HEAD

mkdir -p logs

echo "== GPU STATUS =="
sinfo -o "%20P %18G %8D %8t %10C %10m %N" | grep -Ei 'gpu|ckpt|h200|a100|a40|l40|rtx6k' || true
squeue -u "$USER" || true

echo "== SUBMIT TINY =="
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
  echo "Trying: sbatch $args slurm/run_tiny.sbatch"
  set +e
  submit_output=$(sbatch $args slurm/run_tiny.sbatch 2>&1)
  submit_status=$?
  set -e
  echo "$submit_output"
  if [ "$submit_status" -eq 0 ]; then
    break
  fi
done

if [ "$submit_status" -ne 0 ]; then
  echo "All tiny submit attempts failed."
  exit 1
fi

JOB_ID=$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/ {print $4}' | tail -1)
if [ -z "$JOB_ID" ]; then
  echo "Could not parse job id from sbatch output."
  exit 1
fi
echo "JOB_ID=$JOB_ID"
JOB_LOG="logs/wine-tiny-${JOB_ID}.out"

echo "== MONITOR =="
while squeue -j "$JOB_ID" -h >/dev/null 2>&1 && [ -n "$(squeue -j "$JOB_ID" -h)" ]; do
  date
  squeue -j "$JOB_ID" || true
  if [ -f "$JOB_LOG" ]; then
    echo "--- tail $JOB_LOG ---"
    tail -120 "$JOB_LOG" || true
  else
    echo "$JOB_LOG not created yet"
  fi
  sleep 60
done

echo "== FINAL STATUS =="
sacct -j "$JOB_ID" --format=JobID,JobName%25,Partition,State,ExitCode,Elapsed,MaxRSS -P 2>/dev/null || true
if [ -f "$JOB_LOG" ]; then
  echo "--- final tail $JOB_LOG ---"
  tail -220 "$JOB_LOG"
else
  echo "$JOB_LOG missing"
  exit 1
fi

echo "== OUTPUTS =="
find artifacts/tiny -maxdepth 4 -type f 2>/dev/null | sort || true

echo "== VALIDATE OUTPUTS =="
.venv-hyak/bin/python - <<'PY'
from pathlib import Path

import numpy as np
import pandas as pd

required_files = [
    Path("artifacts/tiny/mse/predictions.parquet"),
    Path("artifacts/tiny/var/predictions.parquet"),
    Path("artifacts/tiny/summary/metrics.csv"),
]
missing = [str(path) for path in required_files if not path.exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))

required_prediction_columns = {"sample_id", "split_role", "y_true", "pred_scaled", "pred_mean"}
for path in required_files[:2]:
    df = pd.read_parquet(path)
    missing_cols = required_prediction_columns.difference(df.columns)
    if missing_cols:
        raise SystemExit(f"{path} missing columns: {sorted(missing_cols)}")
    if df.empty:
        raise SystemExit(f"{path} is empty")
    print(path, "rows=", len(df), "columns=", ",".join(df.columns))

metrics = pd.read_csv(required_files[2])
if metrics.empty:
    raise SystemExit("metrics.csv is empty")
required_metric_columns = {"method", "estimate", "estimated_variance", "standard_error", "bias", "rmse", "ci_length", "sample_savings"}
missing_metric_cols = required_metric_columns.difference(metrics.columns)
if missing_metric_cols:
    raise SystemExit("metrics.csv missing columns: " + ", ".join(sorted(missing_metric_cols)))

methods = set(metrics["method"].astype(str))
required_methods = {
    "sample_mean",
    "lora_mse+ppi",
    "lora_mse+ppi_plus",
    "lora_var+ppi",
    "lora_var+ppi_plus",
}
missing_methods = required_methods.difference(methods)
if missing_methods:
    raise SystemExit("metrics.csv missing methods: " + ", ".join(sorted(missing_methods)))

if "lambda" in metrics.columns:
    lambda_values = metrics.loc[metrics["method"].astype(str).str.contains("ppi_plus"), "lambda"].dropna()
    if not lambda_values.empty and not np.isfinite(lambda_values.to_numpy(dtype=float)).all():
        raise SystemExit("non-finite lambda in ppi_plus rows")

print("metrics rows=", len(metrics))
print(metrics[["method", "estimate", "bias", "rmse", "estimated_variance", "standard_error", "ci_length", "sample_savings"]])
PY

echo "tiny_e2e_task_done"
