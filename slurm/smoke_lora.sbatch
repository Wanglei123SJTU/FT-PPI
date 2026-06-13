#!/bin/bash
#SBATCH --job-name=wine-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

python -m src.train.train_regression --config configs/smoke.yaml --loss mse
python -m src.train.train_regression --config configs/smoke.yaml --loss var

