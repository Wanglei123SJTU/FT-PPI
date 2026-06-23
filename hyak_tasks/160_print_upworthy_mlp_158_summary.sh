#!/bin/bash
set -euo pipefail

echo "upw160_print_summary_task_start $(date)"

cd ~/FT-PPI

TRACE_DIR="artifacts/upworthy_m_estimation/formal_trace_embedding_mlp_158"
NUMERIC_DIR="artifacts/upworthy_m_estimation/formal_numeric_embedding_mlp_158"

for DIR in "$TRACE_DIR" "$NUMERIC_DIR"; do
  echo "SUMMARY_DIR $DIR"
  if [[ ! -d "$DIR" ]]; then
    echo "missing_dir $DIR"
    continue
  fi
  for FILE in target_diagnostics.csv embedding_mlp_summary.csv budget_win_summary.csv; do
    PATH_TO_FILE="$DIR/$FILE"
    echo "BEGIN_FILE $PATH_TO_FILE"
    if [[ -f "$PATH_TO_FILE" ]]; then
      cat "$PATH_TO_FILE"
    else
      echo "missing_file $PATH_TO_FILE"
    fi
    echo "END_FILE $PATH_TO_FILE"
  done
done

echo "upw160_print_summary_task_done $(date)"
