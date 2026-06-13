#!/bin/bash
set -euo pipefail

echo "recover_zerohead_confirmation_task_start"
hostname
date
git rev-parse --short HEAD

JOB_ID="${ZEROHEAD_CONFIRMATION_JOB_ID:-36077278}"
JOB_LOG="logs/wine-zconfirm-${JOB_ID}.out"
ROOT="artifacts/zerohead_allocation_confirmation"

echo "recovering JOB_ID=$JOB_ID"
squeue -j "$JOB_ID" || true

echo "== MONITOR EXISTING JOB =="
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
fi

echo "== VALIDATE OUTPUTS =="
.venv-hyak/bin/python - <<'PY'
from pathlib import Path

import numpy as np
import pandas as pd

root = Path("artifacts/zerohead_allocation_confirmation")
summary = root / "summary"
required = [
    summary / "allocation_metrics.csv",
    summary / "allocation_curve.csv",
    summary / "prediction_diagnostics.csv",
    summary / "oracle_allocations.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))

metrics = pd.read_csv(summary / "allocation_metrics.csv")
curve = pd.read_csv(summary / "allocation_curve.csv")
diagnostics = pd.read_csv(summary / "prediction_diagnostics.csv")
oracle = pd.read_csv(summary / "oracle_allocations.csv")
if metrics.empty or curve.empty or diagnostics.empty or oracle.empty:
    raise SystemExit("summary tables are empty")
if len(metrics) != 4 * 3 * 3:
    raise SystemExit(f"unexpected metric row count: {len(metrics)}")

ppi_plus = curve[curve["method"].astype(str).str.contains("ppi_plus", regex=False)].copy()
if ppi_plus.empty:
    raise SystemExit("missing ppi_plus rows")
if not np.isfinite(ppi_plus["normalized_estimated_variance"]).all():
    raise SystemExit("non-finite normalized variance")
if set(ppi_plus["n_replications"].astype(int)) != {3}:
    raise SystemExit("not all ppi_plus rows have 3 replications")

best = ppi_plus.sort_values("normalized_estimated_variance").iloc[0]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
print("zero-head allocation confirmation metrics rows=", len(metrics))
print("zero-head allocation confirmation ppi_plus")
print(ppi_plus.to_string(index=False))
print("zero-head allocation confirmation oracle")
print(oracle.to_string(index=False))
print(
    "zero-head allocation confirmation best",
    f"train_size={int(best['train_size'])}",
    f"ratio={float(best['allocation_ratio']):.4f}",
    f"normalized_variance={float(best['normalized_estimated_variance']):.6f}",
    f"estimated_variance={float(best['mean_estimated_variance']):.6f}",
    flush=True,
)
print("zero-head allocation confirmation prediction diagnostics")
print(
    diagnostics[diagnostics["role"].isin(["population", "correction"])]
    .sort_values(["role", "train_size"])
    .to_string(index=False)
)
PY

echo "recover_zerohead_confirmation_task_done"
