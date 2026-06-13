#!/bin/bash
set -euo pipefail

echo "zerohead_allocation_probe_task_start"
hostname
date
git rev-parse --short HEAD

mkdir -p logs

echo "== GPU STATUS =="
sinfo -o "%20P %18G %8D %8t %10C %10m %N" | grep -Ei 'gpu|ckpt|h200|a100|a40|l40|rtx6k' || true
squeue -u "$USER" || true

echo "== SUBMIT ZERO-HEAD ALLOCATION PROBE =="
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
  echo "Trying: sbatch $args slurm/run_zerohead_allocation_probe.sbatch"
  set +e
  submit_output=$(sbatch $args slurm/run_zerohead_allocation_probe.sbatch 2>&1)
  submit_status=$?
  set -e
  echo "$submit_output"
  if [ "$submit_status" -eq 0 ]; then
    break
  fi
done

if [ "$submit_status" -ne 0 ]; then
  echo "All zero-head allocation probe submit attempts failed."
  exit 1
fi

JOB_ID=$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/ {print $4}' | tail -1)
if [ -z "$JOB_ID" ]; then
  echo "Could not parse job id from sbatch output."
  exit 1
fi
echo "JOB_ID=$JOB_ID"
JOB_LOG="logs/wine-zalloc-${JOB_ID}.out"

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
  tail -320 "$JOB_LOG"
else
  echo "$JOB_LOG missing"
  exit 1
fi

echo "== VALIDATE OUTPUTS =="
.venv-hyak/bin/python - <<'PY'
from pathlib import Path

import pandas as pd

root = Path("artifacts/zerohead_allocation_probe")
summary = root / "summary"
required = [
    summary / "allocation_metrics.csv",
    summary / "allocation_curve.csv",
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
if len(metrics) != 5 * 3:
    raise SystemExit(f"unexpected metric row count: {len(metrics)}")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
print("zero-head allocation metrics rows=", len(metrics))
ppi = curve[curve["method"].astype(str).str.contains("ppi_plus", regex=False)]
print("zero-head allocation ppi_plus")
print(ppi.to_string(index=False))
print("zero-head allocation prediction diagnostics")
print(
    diagnostics[diagnostics["role"].isin(["population", "correction"])]
    .sort_values(["role", "train_size"])
    .to_string(index=False)
)
PY

echo "zerohead_allocation_probe_task_done"
