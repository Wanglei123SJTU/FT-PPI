#!/bin/bash
set -euo pipefail

echo "first_pilot_task_start"
hostname
date
git rev-parse --short HEAD

mkdir -p logs

echo "== GPU STATUS =="
sinfo -o "%20P %18G %8D %8t %10C %10m %N" | grep -Ei 'gpu|ckpt|h200|a100|a40|l40|rtx6k' || true
squeue -u "$USER" || true

echo "== SUBMIT FIRST PILOT =="
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
  echo "Trying: sbatch $args slurm/run_first_pilot.sbatch"
  set +e
  submit_output=$(sbatch $args slurm/run_first_pilot.sbatch 2>&1)
  submit_status=$?
  set -e
  echo "$submit_output"
  if [ "$submit_status" -eq 0 ]; then
    break
  fi
done

if [ "$submit_status" -ne 0 ]; then
  echo "All first pilot submit attempts failed."
  exit 1
fi

JOB_ID=$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/ {print $4}' | tail -1)
if [ -z "$JOB_ID" ]; then
  echo "Could not parse job id from sbatch output."
  exit 1
fi
echo "JOB_ID=$JOB_ID"
JOB_LOG="logs/wine-pilot-${JOB_ID}.out"

echo "== MONITOR =="
while squeue -j "$JOB_ID" -h >/dev/null 2>&1 && [ -n "$(squeue -j "$JOB_ID" -h)" ]; do
  date
  squeue -j "$JOB_ID" || true
  if [ -f "$JOB_LOG" ]; then
    echo "--- tail $JOB_LOG ---"
    tail -160 "$JOB_LOG" || true
  else
    echo "$JOB_LOG not created yet"
  fi
  sleep 90
done

echo "== FINAL STATUS =="
sacct -j "$JOB_ID" --format=JobID,JobName%25,Partition,State,ExitCode,Elapsed,MaxRSS -P 2>/dev/null || true
if [ -f "$JOB_LOG" ]; then
  echo "--- final tail $JOB_LOG ---"
  tail -260 "$JOB_LOG"
else
  echo "$JOB_LOG missing"
  exit 1
fi

echo "== OUTPUTS =="
find artifacts/first_pilot -maxdepth 5 -type f 2>/dev/null | sort || true

echo "== VALIDATE OUTPUTS =="
.venv-hyak/bin/python - <<'PY'
from pathlib import Path

import numpy as np
import pandas as pd

root = Path("artifacts/first_pilot")
summary = root / "summary" / "diagnostic_summary.csv"
metrics = root / "summary" / "allocation_metrics.csv"
required_files = [summary, metrics]
missing = [str(path) for path in required_files if not path.exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))

diagnostics = pd.read_csv(summary)
all_metrics = pd.read_csv(metrics)
if diagnostics.empty or all_metrics.empty:
    raise SystemExit("first pilot diagnostics are empty")

expected_tags = ["B0500_s0050_v0100", "B0500_s0100_v0100", "B1000_s0100_v0100", "B1000_s0200_v0100"]
for tag in expected_tags:
    for loss in ["mse", "var"]:
        pred = root / tag / loss / "predictions.parquet"
        if not pred.exists():
            raise SystemExit(f"missing prediction file: {pred}")
        df = pd.read_parquet(pred)
        required_prediction_columns = {"sample_id", "split_role", "y_true", "pred_scaled", "pred_mean"}
        missing_pred_cols = required_prediction_columns.difference(df.columns)
        if missing_pred_cols:
            raise SystemExit(f"{pred} missing columns: {sorted(missing_pred_cols)}")
        if len(df) != 20000:
            raise SystemExit(f"{pred} has {len(df)} rows; expected 20000")

expected_methods = {
    "sample_mean",
    "lora_mse+ppi",
    "lora_mse+ppi_plus",
    "lora_var+ppi",
    "lora_var+ppi_plus",
}
observed_methods = set(all_metrics["method"].astype(str))
missing_methods = expected_methods.difference(observed_methods)
if missing_methods:
    raise SystemExit("missing methods: " + ", ".join(sorted(missing_methods)))

expected_budgets = {500, 1000}
if set(diagnostics["budget"].astype(int)) != expected_budgets:
    raise SystemExit("unexpected budgets in diagnostics")

finite_cols = ["estimated_estimator_variance", "standard_error", "ci_length"]
if not np.isfinite(diagnostics[finite_cols].to_numpy(dtype=float)).all():
    raise SystemExit("non-finite diagnostic values")

print("first pilot diagnostic rows=", len(diagnostics))
print(diagnostics[["budget", "allocation_ratio", "train_size", "loss", "method", "residual_variance", "estimated_estimator_variance", "ci_length"]])
PY

echo "first_pilot_task_done"
