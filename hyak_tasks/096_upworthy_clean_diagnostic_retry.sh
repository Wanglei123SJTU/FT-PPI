#!/bin/bash
set -euo pipefail

echo "upworthy_clean_diagnostic_retry_task_start $(date)"
hostname
git rev-parse --short HEAD

echo "== 095 MARKER STATUS =="
for marker_dir in .hyak_runner/done .hyak_runner/failed .hyak_runner/running; do
  marker="$marker_dir/095_upworthy_clean_diagnostic"
  if [ -e "$marker" ]; then
    echo "-- $marker --"
    cat "$marker" || true
  else
    echo "missing $marker"
  fi
done

echo "== RUN CLEAN DIAGNOSTIC VIA 095 SCRIPT =="
bash hyak_tasks/095_upworthy_clean_diagnostic.sh

echo "upworthy_clean_diagnostic_retry_task_done $(date)"
