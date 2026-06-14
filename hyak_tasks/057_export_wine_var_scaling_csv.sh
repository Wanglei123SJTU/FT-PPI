#!/bin/bash
set -euo pipefail

ROOT="/gscratch/scrubbed/lei0603/ft-ppi/artifacts/wine_var_scaling_b10000"

echo "wine_var_scaling_csv_export_start"
date
git rev-parse --short HEAD

for path in \
  "$ROOT/training_runtime_summary.csv" \
  "$ROOT/scaling_fit_by_rep.csv" \
  "$ROOT/rampup_recovery_by_rep.csv"
do
  if [ ! -s "$path" ]; then
    echo "missing_or_empty $path" >&2
    exit 1
  fi
done

echo "== TRAINING_RUNTIME_SUMMARY_CSV_BEGIN =="
cat "$ROOT/training_runtime_summary.csv"
echo "== TRAINING_RUNTIME_SUMMARY_CSV_END =="

echo "== SCALING_FIT_BY_REP_CSV_BEGIN =="
cat "$ROOT/scaling_fit_by_rep.csv"
echo "== SCALING_FIT_BY_REP_CSV_END =="

echo "== RAMPUP_RECOVERY_BY_REP_CSV_BEGIN =="
cat "$ROOT/rampup_recovery_by_rep.csv"
echo "== RAMPUP_RECOVERY_BY_REP_CSV_END =="

echo "wine_var_scaling_csv_export_done"
