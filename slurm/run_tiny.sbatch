#!/bin/bash
#SBATCH --job-name=wine-tiny
#SBATCH --partition=ckpt
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load coenv/python/3.11.9
module load cuda/12.4.1

VENV_DIR="${HYAK_VENV_DIR:-.venv-hyak}"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "Missing $VENV_DIR. Run: bash scripts/setup_hyak_env.sh"
  exit 1
fi

source "$VENV_DIR/bin/activate"

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
