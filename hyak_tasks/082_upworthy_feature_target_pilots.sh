#!/bin/bash
set -euo pipefail

echo "upworthy_feature_target_pilots_task_start"
hostname
date
git rev-parse --short HEAD

run_pilot() {
  local config="$1"
  local array_conc="$2"
  local min_idle="$3"
  echo "== RUN PILOT config=${config} array_conc=${array_conc} min_idle=${min_idle} =="
  export CONFIG="$config"
  export ARRAY_CONC="$array_conc"
  export HYAK_GPU_MIN_IDLE="$min_idle"
  unset HYAK_FORCE_GPU_ARGS
  bash hyak_tasks/080_upworthy_question_scaling_diagnostic_full.sh
}

run_pilot configs/upworthy_length_single_scaling_pilot_1p5b.yaml 8 8
run_pilot configs/upworthy_simplicity_single_scaling_pilot_1p5b.yaml 8 8
run_pilot configs/upworthy_length_full5_scaling_pilot_1p5b.yaml 8 8
run_pilot configs/upworthy_length_single_scaling_pilot_7b.yaml 2 2

echo "upworthy_feature_target_pilots_task_done"
