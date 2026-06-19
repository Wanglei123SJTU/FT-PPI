#!/bin/bash
set -euo pipefail

echo "upworthy_targeted_ifvar_fast_screen_task_start"
hostname
date
git rev-parse --short HEAD

run_target() {
  local config="$1"
  echo "== RUN TARGETED FAST IFVAR config=${config} =="
  export CONFIG="$config"
  export ARRAY_CONC="${ARRAY_CONC:-8}"
  export HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-8}"
  unset HYAK_FORCE_GPU_ARGS
  bash hyak_tasks/080_upworthy_question_scaling_diagnostic_full.sh
}

run_target configs/upworthy_question_full5_textonly_fast_ifvar_1p5b.yaml
run_target configs/upworthy_numeric_full5_textonly_fast_ifvar_1p5b.yaml
run_target configs/upworthy_simplicity_full5_textonly_fast_ifvar_1p5b.yaml
run_target configs/upworthy_common_full5_textonly_fast_ifvar_1p5b.yaml
run_target configs/upworthy_length_full5_textonly_fast_ifvar_1p5b.yaml

echo "upworthy_targeted_ifvar_fast_screen_task_done"
