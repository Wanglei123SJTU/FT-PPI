#!/bin/bash
set -euo pipefail

echo "cancel_pending_wine_theory_jobs_before_l40s_retry"
old_jobs="$(squeue -u "$USER" -n wine-theory -h -o "%A" 2>/dev/null | sort -u || true)"
if [ -n "$old_jobs" ]; then
  echo "$old_jobs" | xargs -r scancel
  echo "cancelled_wine_theory_jobs=$old_jobs"
else
  echo "no_existing_wine_theory_jobs"
fi

export HYAK_FORCE_GPU_ARGS="${HYAK_FORCE_GPU_ARGS:---partition=ckpt-g2 --gres=gpu:l40s:1}"
exec bash hyak_tasks/074_wine_full_grid_theory_exact_h200.sh
