#!/bin/bash
set -euo pipefail

echo "print_loss_controlled_r10_results_task_start"
ROOT="/gscratch/scrubbed/$USER/ft-ppi/artifacts/wine_loss_controlled_b10000_r10_stopmetric_s50"

if [ ! -d "$ROOT" ]; then
  echo "missing output directory: $ROOT"
  exit 1
fi

for name in \
  training_runtime_summary.csv \
  loss_comparison_summary.csv \
  rampup_recovery_summary.csv \
  scaling_fit_by_rep.csv \
  leakage_check_report.json
do
  path="$ROOT/$name"
  if [ ! -s "$path" ]; then
    echo "missing or empty: $path"
    exit 1
  fi
  echo "BEGIN_LOSS_CONTROLLED_R10_FILE $name"
  cat "$path"
  echo "END_LOSS_CONTROLLED_R10_FILE $name"
done

echo "print_loss_controlled_r10_results_task_done"
