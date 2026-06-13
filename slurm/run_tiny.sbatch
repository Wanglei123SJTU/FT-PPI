#!/bin/bash
#SBATCH --job-name=wine-tiny
#SBATCH --partition=ckpt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load cuda/12.4.1
module load foster/python/miniconda/3.8 2>/dev/null || \
  module load chem/miniconda3/3.8 2>/dev/null || \
  module load coenv/miniconda/3.13.11 2>/dev/null || \
  module load coenv/python/3.11.9

VENV_DIR="${HYAK_VENV_DIR:-.venv-hyak}"
VENV_REAL="$(readlink -f "$VENV_DIR" 2>/dev/null || echo "$VENV_DIR")"
CACHE_ROOT="${HYAK_CACHE_DIR:-$(dirname "$VENV_REAL")/cache}"
mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/hf_datasets" "$CACHE_ROOT/torch"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/hf_datasets}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "Missing $VENV_DIR. Run: bash scripts/setup_hyak_env.sh"
  exit 1
fi

if [ -d "$VENV_DIR/conda-meta" ] && command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$VENV_DIR"
else
  source "$VENV_DIR/bin/activate"
fi

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

python -m src.train.train_regression --config configs/tiny.yaml --loss mse
python -m src.train.train_regression --config configs/tiny.yaml --loss var
python -m src.analysis.summarize \
  --population artifacts/tiny/mse/population.csv \
  --predictions artifacts/tiny/mse/predictions.parquet artifacts/tiny/var/predictions.parquet \
  --output-dir artifacts/tiny/summary
