#!/bin/bash
set -euo pipefail

echo "upworthy_working_method_pilots_task_start"
hostname
date
git rev-parse --short HEAD

run_pilot() {
  local config="$1"
  local array_conc="$2"
  echo "== RUN WORKING-METHOD PILOT config=${config} array_conc=${array_conc} =="
  export CONFIG="$config"
  export ARRAY_CONC="$array_conc"
  unset HYAK_FORCE_GPU_ARGS
  bash hyak_tasks/080_upworthy_question_scaling_diagnostic_full.sh
}

run_pilot configs/upworthy_numeric_single_scaling_pilot_1p5b.yaml 8
run_pilot configs/upworthy_length_single_linear_head_scaling_pilot_1p5b.yaml 8
run_pilot configs/upworthy_length_single_precision_weighted_scaling_pilot_1p5b.yaml 8
run_pilot configs/upworthy_length_single_ctr_scaling_pilot_1p5b.yaml 8

echo "upworthy_working_method_pilots_task_done"
