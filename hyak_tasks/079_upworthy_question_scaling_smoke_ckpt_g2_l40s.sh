#!/bin/bash
set -euo pipefail

export HYAK_FORCE_GPU_ARGS="${HYAK_FORCE_GPU_ARGS:---partition=ckpt-g2 --gres=gpu:l40s:1}"

bash hyak_tasks/078_upworthy_question_scaling_smoke.sh
