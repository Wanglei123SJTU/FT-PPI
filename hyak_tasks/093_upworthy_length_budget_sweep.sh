#!/bin/bash
set -euo pipefail

echo "upworthy_length_budget_sweep_task_start"
hostname
date
git rev-parse --short HEAD

export CONFIG="configs/upworthy_length_full5_textonly_budget_sweep_1p5b.yaml"
export ARRAY_CONC="${ARRAY_CONC:-8}"
export HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-8}"
unset HYAK_FORCE_GPU_ARGS

bash hyak_tasks/080_upworthy_question_scaling_diagnostic_full.sh

echo "upworthy_length_budget_sweep_task_done"
