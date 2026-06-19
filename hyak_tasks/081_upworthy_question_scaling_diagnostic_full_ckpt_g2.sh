#!/bin/bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/upworthy_question_scaling_diagnostic_full.yaml}"
export ARRAY_CONC="${ARRAY_CONC:-8}"
export HYAK_FORCE_GPU_ARGS="${HYAK_FORCE_GPU_ARGS:---partition=ckpt-g2 --gres=gpu:1}"

bash hyak_tasks/080_upworthy_question_scaling_diagnostic_full.sh
