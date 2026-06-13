#!/bin/bash
set -euo pipefail

echo "allocation_regret_analysis_task_start"
hostname
date
git rev-parse --short HEAD

for root in \
  artifacts/allocation_scaling_probe \
  artifacts/zerohead_allocation_probe \
  artifacts/zerohead_allocation_confirmation
do
  summary="$root/summary"
  echo "== RAMP-UP REGRET: $root =="
  if [ ! -f "$summary/allocation_curve.csv" ]; then
    echo "missing $summary/allocation_curve.csv"
    continue
  fi
  .venv-hyak/bin/python -m src.analysis.allocation_regret --summary-dir "$summary" --rampup-points 3
done

echo "allocation_regret_analysis_task_done"
