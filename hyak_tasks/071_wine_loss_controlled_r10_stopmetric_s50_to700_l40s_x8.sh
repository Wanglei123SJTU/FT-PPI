#!/bin/bash
set -euo pipefail

echo "wine_loss_controlled_r10_stopmetric_s50_to700_l40s_x8_task_start"
export HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-16}"
export CONFIG="configs/wine_loss_controlled_b10000_r10_stopmetric_s50_to700.yaml"
export SLURM_TEMPLATE="slurm/run_wine_loss_controlled_b10000_r10_stopmetric_s50_to700.sbatch"

bash hyak_tasks/060_wine_loss_controlled_gpu_fallback.sh

ROOT="/gscratch/scrubbed/$USER/ft-ppi/artifacts/wine_loss_controlled_b10000_r10_stopmetric_s50_to700"
echo "== POST-RUN VISUALIZATION =="
.venv-hyak/bin/python -m src.analysis.plot_loss_scaling --input-dir "$ROOT" --primary-method var_stop_var

echo "== SAVED RESULT FILES =="
find "$ROOT" -maxdepth 2 -type f \( -name "*.csv" -o -name "*.json" -o -name "*.pdf" -o -name "*.png" \) -printf "%s %p\n" | sort -n

echo "BEGIN_LOSS_CONTROLLED_TO700_RUNTIME_SUMMARY_CSV"
cat "$ROOT/training_runtime_summary.csv"
echo "END_LOSS_CONTROLLED_TO700_RUNTIME_SUMMARY_CSV"

echo "BEGIN_LOSS_CONTROLLED_TO700_SCALING_FIT_BY_REP_CSV"
cat "$ROOT/scaling_fit_by_rep.csv"
echo "END_LOSS_CONTROLLED_TO700_SCALING_FIT_BY_REP_CSV"

echo "BEGIN_LOSS_CONTROLLED_TO700_RAMPUP_RECOVERY_BY_REP_CSV"
cat "$ROOT/rampup_recovery_by_rep.csv"
echo "END_LOSS_CONTROLLED_TO700_RAMPUP_RECOVERY_BY_REP_CSV"

echo "wine_loss_controlled_r10_stopmetric_s50_to700_l40s_x8_task_done"
