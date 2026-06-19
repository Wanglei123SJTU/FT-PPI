#!/bin/bash
set -euo pipefail

echo "upworthy_feature_target_pilots_ckptg2_rerun_task_start"
hostname
date
git rev-parse --short HEAD

OLD_JOB_ID="${OLD_JOB_ID:-36185322}"
echo "cancel_old_job=${OLD_JOB_ID}"
scancel "$OLD_JOB_ID" 2>/dev/null || true

run_pilot() {
  local config="$1"
  local array_conc="$2"
  echo "== RUN CKPT-G2 PILOT config=${config} array_conc=${array_conc} =="
  export CONFIG="$config"
  export ARRAY_CONC="$array_conc"
  export HYAK_FORCE_GPU_ARGS="--partition=ckpt-g2 --gres=gpu:1"
  bash hyak_tasks/080_upworthy_question_scaling_diagnostic_full.sh
}

run_pilot configs/upworthy_length_single_scaling_pilot_1p5b.yaml 8
run_pilot configs/upworthy_simplicity_single_scaling_pilot_1p5b.yaml 8
run_pilot configs/upworthy_length_full5_scaling_pilot_1p5b.yaml 8
run_pilot configs/upworthy_length_single_scaling_pilot_7b.yaml 2

echo "upworthy_feature_target_pilots_ckptg2_rerun_task_done"
