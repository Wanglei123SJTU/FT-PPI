#!/bin/bash
set -euo pipefail

echo "allocation_scaling_probe_task_start"
hostname
date
git rev-parse --short HEAD

mkdir -p logs

echo "== GPU STATUS =="
sinfo -o "%20P %18G %8D %8t %10C %10m %N" | grep -Ei 'gpu|ckpt|h200|a100|a40|l40|rtx6k' || true
squeue -u "$USER" || true

echo "== SUBMIT ALLOCATION SCALING PROBE =="
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
  echo "Trying: sbatch $args slurm/run_allocation_scaling_probe.sbatch"
  set +e
  submit_output=$(sbatch $args slurm/run_allocation_scaling_probe.sbatch 2>&1)
  submit_status=$?
  set -e
  echo "$submit_output"
  if [ "$submit_status" -eq 0 ]; then
    break
  fi
done

if [ "$submit_status" -ne 0 ]; then
  echo "All allocation scaling probe submit attempts failed."
  exit 1
fi

JOB_ID=$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/ {print $4}' | tail -1)
if [ -z "$JOB_ID" ]; then
  echo "Could not parse job id from sbatch output."
  exit 1
fi
echo "JOB_ID=$JOB_ID"
JOB_LOG="logs/wine-scale-${JOB_ID}.out"

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
  sleep 180
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
find artifacts/allocation_scaling_probe -maxdepth 4 -type f 2>/dev/null | sort || true

echo "== VALIDATE OUTPUTS =="
.venv-hyak/bin/python - <<'PY'
from pathlib import Path

import numpy as np
import pandas as pd

root = Path("artifacts/allocation_scaling_probe")
summary = root / "summary"
required = [
    summary / "allocation_metrics.csv",
    summary / "diagnostic_summary.csv",
    summary / "allocation_curve.csv",
    summary / "oracle_allocations.csv",
    summary / "scaling_law_fit.csv",
    summary / "scaling_law_loso.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))

metrics = pd.read_csv(summary / "allocation_metrics.csv")
curve = pd.read_csv(summary / "allocation_curve.csv")
oracle = pd.read_csv(summary / "oracle_allocations.csv")
scaling = pd.read_csv(summary / "scaling_law_fit.csv")
loso = pd.read_csv(summary / "scaling_law_loso.csv")
if metrics.empty or curve.empty or oracle.empty or scaling.empty or loso.empty:
    raise SystemExit("one or more summary tables are empty")

expected_reps = {0, 1, 2}
if set(metrics["replication_id"].astype(int)) != expected_reps:
    raise SystemExit("unexpected replications in metrics")
expected_train = {50, 75, 100, 125, 150, 200, 300}
if not expected_train.issubset(set(metrics["train_size"].astype(int))):
    raise SystemExit("missing train sizes")
if set(metrics["budget"].astype(int)) != {1000}:
    raise SystemExit("unexpected budgets")
if len(metrics) != 3 * 7 * 6:
    raise SystemExit(f"unexpected metric row count: {len(metrics)}")

finite_cols = ["mean_estimated_variance", "normalized_estimated_variance"]
if not np.isfinite(curve[finite_cols].to_numpy(dtype=float)).all():
    raise SystemExit("non-finite allocation curve values")
if not set(scaling["loss"].astype(str)).issuperset({"mse", "var"}):
    raise SystemExit("scaling fit missing losses")

print("allocation scaling metrics rows=", len(metrics))
print("oracle allocations")
print(oracle)
print("scaling law fit")
print(scaling)
PY

echo "allocation_scaling_probe_task_done"
