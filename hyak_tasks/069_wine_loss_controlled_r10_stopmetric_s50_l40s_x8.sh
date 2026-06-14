#!/bin/bash
set -euo pipefail

echo "wine_loss_controlled_r10_stopmetric_s50_l40s_x8_task_start"
export HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-16}"
export CONFIG="configs/wine_loss_controlled_b10000_r10_stopmetric_s50.yaml"
export SLURM_TEMPLATE="slurm/run_wine_loss_controlled_b10000_r10_stopmetric_s50.sbatch"
bash hyak_tasks/060_wine_loss_controlled_gpu_fallback.sh
echo "wine_loss_controlled_r10_stopmetric_s50_l40s_x8_task_done"
