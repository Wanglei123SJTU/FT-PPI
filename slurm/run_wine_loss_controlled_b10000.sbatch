#!/bin/bash
#SBATCH --job-name=wine-lossctrl
#SBATCH --partition=ckpt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-59%8
#SBATCH --output=/gscratch/scrubbed/lei0603/ft-ppi/logs/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
mkdir -p /gscratch/scrubbed/lei0603/ft-ppi/logs

module purge
module load cuda/12.4.1
module load foster/python/miniconda/3.8 2>/dev/null || \
  module load chem/miniconda3/3.8 2>/dev/null || \
  module load coenv/miniconda/3.13.11 2>/dev/null || \
  module load coenv/python/3.11.9

VENV_DIR="${HYAK_VENV_DIR:-.venv-hyak}"
VENV_REAL="$(readlink -f "$VENV_DIR" 2>/dev/null || echo "$VENV_DIR")"
CACHE_ROOT="${HYAK_CACHE_DIR:-$(dirname "$VENV_REAL")/cache}"
mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/hf_datasets" "$CACHE_ROOT/torch" "$CACHE_ROOT/conda_pkgs"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/hf_datasets}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$CACHE_ROOT/conda_pkgs}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Missing $VENV_DIR. Run: bash scripts/setup_hyak_env.sh"
  exit 1
fi

if [ -d "$VENV_DIR/conda-meta" ] && command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$VENV_REAL"
else
  source "$VENV_DIR/bin/activate"
fi

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("bf16_supported", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY

CONFIG="${CONFIG:-configs/wine_loss_controlled_b10000.yaml}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
echo "wine_loss_controlled_cell_start task_index=${TASK_ID} config=${CONFIG}"
python -m src.experiments.wine_var_scaling_law train-cell --config "$CONFIG" --task-index "$TASK_ID"
echo "wine_loss_controlled_cell_done task_index=${TASK_ID}"
