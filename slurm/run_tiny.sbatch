#!/bin/bash
#SBATCH --job-name=wine-tiny
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

python -m src.train.train_regression --config configs/tiny.yaml --loss mse
python -m src.train.train_regression --config configs/tiny.yaml --loss var
python -m src.analysis.summarize \
  --population artifacts/tiny/mse/population.csv \
  --predictions artifacts/tiny/mse/predictions.parquet artifacts/tiny/var/predictions.parquet \
  --output-dir artifacts/tiny/summary

