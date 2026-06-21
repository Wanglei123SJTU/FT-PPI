#!/bin/bash
set -euo pipefail

echo "upworthy_clean_diagnostic_retry2_task_start $(date)"
hostname
git rev-parse --short HEAD

echo "== PREVIOUS MARKER STATUS =="
for task_id in 095_upworthy_clean_diagnostic 096_upworthy_clean_diagnostic_retry; do
  for marker_dir in .hyak_runner/done .hyak_runner/failed .hyak_runner/running; do
    marker="$marker_dir/$task_id"
    if [ -e "$marker" ]; then
      echo "-- $marker --"
      cat "$marker" || true
    else
      echo "missing $marker"
    fi
  done
done

echo "== RUN CLEAN DIAGNOSTIC VIA FIXED 095 SCRIPT =="
bash hyak_tasks/095_upworthy_clean_diagnostic.sh

echo "upworthy_clean_diagnostic_retry2_task_done $(date)"
