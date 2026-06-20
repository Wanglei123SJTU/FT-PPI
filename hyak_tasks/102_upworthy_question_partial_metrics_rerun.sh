#!/bin/bash
set -euo pipefail

TASK_ID="102_upworthy_question_partial_metrics_rerun"
LOCK_DIR=".hyak_runner/task_locks"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/${TASK_ID}.lock"
if ! flock -n 9; then
  echo "${TASK_ID}_already_running_elsewhere"
  exit 0
fi

echo "${TASK_ID}_task_start"
hostname
date
git rev-parse --short HEAD

bash hyak_tasks/101_upworthy_question_partial_metrics.sh

echo "${TASK_ID}_task_done"
