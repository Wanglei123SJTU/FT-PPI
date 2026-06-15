#!/bin/bash
set -euo pipefail

export HYAK_FORCE_GPU_ARGS="${HYAK_FORCE_GPU_ARGS:---partition=ckpt-g2 --gres=gpu:h200:1}"
exec bash hyak_tasks/074_wine_full_grid_theory_exact_h200.sh
