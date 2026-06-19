#!/bin/bash
set -euo pipefail

echo "upworthy_textonly_surrogate_pilots_task_start"
hostname
date
git rev-parse --short HEAD

run_pilot() {
  local config="$1"
  local array_conc="$2"
  echo "== RUN TEXT-ONLY SURROGATE PILOT config=${config} array_conc=${array_conc} =="
  export CONFIG="$config"
  export ARRAY_CONC="$array_conc"
  unset HYAK_FORCE_GPU_ARGS
  bash hyak_tasks/080_upworthy_question_scaling_diagnostic_full.sh
}

run_pilot configs/upworthy_question_full5_textonly_scaling_pilot_1p5b.yaml 8
run_pilot configs/upworthy_length_full5_textonly_scaling_pilot_1p5b.yaml 8

echo "upworthy_textonly_surrogate_pilots_task_done"
