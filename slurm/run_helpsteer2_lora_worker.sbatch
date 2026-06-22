#!/bin/bash
#SBATCH --job-name=hs2-worker
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
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
if [ ! -x "$VENV_DIR/bin/python" ] && [ -x ".venv-hyak/bin/python" ]; then
  VENV_DIR=".venv-hyak"
fi
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
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

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

echo "VENV_DIR=$VENV_DIR"
echo "VENV_REAL=$VENV_REAL"
echo "python=$(command -v python)"
PYTHON_BIN="$VENV_REAL/bin/python"
echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" --version

"$PYTHON_BIN" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("bf16_supported", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY

WORKER_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
NUM_WORKERS="${NUM_WORKERS:?NUM_WORKERS is required}"
INPUT_CSV="${INPUT_CSV:-Data/helpsteer2_preference_pairs.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/gscratch/scrubbed/${USER}/ft-ppi/artifacts/helpsteer2_lora_worker}"
PLAN_CSV="${PLAN_CSV:-$OUTPUT_DIR/cell_plan.csv}"
FEATURES="${FEATURES:-delta_log_length_scale,delta_log_sentences_scale}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
MAX_LENGTH="${MAX_LENGTH:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-4}"
PATIENCE="${PATIENCE:-1}"
VALIDATION_LIMIT="${VALIDATION_LIMIT:-512}"
EVALUATION_LIMIT="${EVALUATION_LIMIT:-1024}"
EXTRA_ARGS=()
if [ "${NO_GRADIENT_CHECKPOINTING:-0}" = "1" ]; then
  EXTRA_ARGS+=(--no-gradient-checkpointing)
fi
if [ "${NO_4BIT:-0}" = "1" ]; then
  EXTRA_ARGS+=(--no-4bit)
fi

echo "helpsteer2_lora_worker_start worker_index=${WORKER_INDEX}/${NUM_WORKERS} output_dir=${OUTPUT_DIR}"
"$PYTHON_BIN" -u -m src.experiments.helpsteer2_lora_scaling \
  --input-csv "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --features "$FEATURES" \
  train-worker \
  --plan-csv "$PLAN_CSV" \
  --worker-index "$WORKER_INDEX" \
  --num-workers "$NUM_WORKERS" \
  --model-name "$MODEL_NAME" \
  --max-length "$MAX_LENGTH" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --max-epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --validation-limit "$VALIDATION_LIMIT" \
  --evaluation-limit "$EVALUATION_LIMIT" \
  "${EXTRA_ARGS[@]}"
echo "helpsteer2_lora_worker_done worker_index=${WORKER_INDEX}/${NUM_WORKERS}"
