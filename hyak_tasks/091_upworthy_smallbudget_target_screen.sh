#!/bin/bash
set -euo pipefail

echo "upworthy_smallbudget_target_screen_task_start"
hostname
date
git rev-parse --short HEAD

run_screen() {
  local config="$1"
  echo "== RUN SMALL-BUDGET TARGET SCREEN config=${config} =="
  export CONFIG="$config"
  export ARRAY_CONC="${ARRAY_CONC:-8}"
  unset HYAK_FORCE_GPU_ARGS
  bash hyak_tasks/080_upworthy_question_scaling_diagnostic_full.sh
}

run_screen configs/upworthy_numeric_full5_textonly_smallbudget_screen_1p5b.yaml
run_screen configs/upworthy_simplicity_full5_textonly_smallbudget_screen_1p5b.yaml
run_screen configs/upworthy_common_full5_textonly_smallbudget_screen_1p5b.yaml

echo "upworthy_smallbudget_target_screen_task_done"
