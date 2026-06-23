#!/bin/bash
set -euo pipefail

echo "upw159_status_task_start $(date)"
echo "host=$(hostname)"
echo "repo=$(pwd)"
echo "head=$(git rev-parse --short HEAD)"

TRACE_JOB=36315508
NUMERIC_JOB=36315509
TRACE_OUT="artifacts/upworthy_m_estimation/formal_trace_embedding_mlp_158"
NUMERIC_OUT="artifacts/upworthy_m_estimation/formal_numeric_embedding_mlp_158"
LOG_ROOT="/gscratch/scrubbed/$USER/ft-ppi/logs"

echo "== SLURM STATUS =="
squeue -j "$TRACE_JOB,$NUMERIC_JOB" || true
echo "== SACCT STATUS =="
sacct -j "$TRACE_JOB,$NUMERIC_JOB" --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,MaxRSS -P 2>/dev/null || true

for spec in "trace:$TRACE_JOB" "numeric:$NUMERIC_JOB"; do
  mode="${spec%%:*}"
  job="${spec##*:}"
  log_file="$LOG_ROOT/upw-mlp-${mode}-${job}.out"
  echo "== LOG TAIL $mode $job =="
  if [ -f "$log_file" ]; then
    tail -120 "$log_file" || true
  else
    echo "missing $log_file"
  fi
done

echo "== OUTPUT FILES =="
find "$TRACE_OUT" "$NUMERIC_OUT" -maxdepth 1 -type f -printf '%p %s bytes\n' 2>/dev/null || true

echo "== OUTPUT SUMMARY =="
if [ -x ".venv-hyak/bin/python" ]; then
  .venv-hyak/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

roots = [
    Path("artifacts/upworthy_m_estimation/formal_trace_embedding_mlp_158"),
    Path("artifacts/upworthy_m_estimation/formal_numeric_embedding_mlp_158"),
]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
for root in roots:
    print(f"SUMMARY_ROOT {root}")
    summary_path = root / "embedding_mlp_summary.csv"
    budget_path = root / "budget_win_summary.csv"
    if not summary_path.exists():
        print(f"missing {summary_path}")
        continue
    summary = pd.read_csv(summary_path)
    print("best variance rows")
    print(
        summary.sort_values(["target", "mean_ifvar_ratio_vs_zero"])
        .groupby("target", as_index=False)
        .head(10)
        .to_string(index=False)
    )
    if budget_path.exists():
        budget = pd.read_csv(budget_path)
        print("best budget rows")
        print(
            budget.sort_values(["target", "budget", "mean_ratio_vs_direct_ols"])
            .groupby(["target", "budget"], as_index=False)
            .head(1)
            .to_string(index=False)
        )
PY
else
  echo "missing .venv-hyak/bin/python"
fi

echo "upw159_status_task_done $(date)"
