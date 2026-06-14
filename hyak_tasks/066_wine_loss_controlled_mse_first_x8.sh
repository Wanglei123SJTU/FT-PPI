#!/bin/bash
set -euo pipefail

echo "wine_loss_controlled_mse_first_x8_task_start"
date

export HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-8}"
bash hyak_tasks/060_wine_loss_controlled_gpu_fallback.sh

echo "wine_loss_controlled_mse_first_x8_task_done"
