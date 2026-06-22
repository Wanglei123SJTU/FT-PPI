#!/bin/bash
set -euo pipefail

echo "helpsteer2_lora_worker_quickdiag_task_start $(date)"

cd "${HYAK_RUNNER_REPO_DIR:-$PWD}"
git status --short --branch

module purge
module load cuda/12.4.1
module load foster/python/miniconda/3.8 2>/dev/null || \
  module load chem/miniconda3/3.8 2>/dev/null || \
  module load coenv/miniconda/3.13.11 2>/dev/null || \
  module load coenv/python/3.11.9

if [ ! -x .venv-hyak/bin/python ]; then
  echo "setting up Hyak Python environment"
  bash scripts/setup_hyak_env.sh
fi

VENV_DIR="${HYAK_VENV_DIR:-.venv-hyak}"
VENV_REAL="$(readlink -f "$VENV_DIR" 2>/dev/null || echo "$VENV_DIR")"
export HYAK_VENV_DIR="$VENV_REAL"
CACHE_ROOT="${HYAK_CACHE_DIR:-$(dirname "$VENV_REAL")/cache}"
mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/hf_datasets" "$CACHE_ROOT/torch" "$CACHE_ROOT/conda_pkgs"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/hf_datasets}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$CACHE_ROOT/conda_pkgs}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [ -d "$VENV_DIR/conda-meta" ] && command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$VENV_REAL"
else
  source "$VENV_DIR/bin/activate"
fi

python -m pytest tests/test_helpsteer2_preference.py -q

INPUT_CSV="Data/helpsteer2_preference_pairs.csv"
if [ ! -s "$INPUT_CSV" ]; then
  echo "preparing HelpSteer2 preference data"
  python -m src.data.prepare_helpsteer2_preference \
    --output-csv "$INPUT_CSV" \
    --summary-json "Data/helpsteer2_preference_pairs.summary.json"
fi

OUTPUT_DIR="${OUTPUT_DIR:-/gscratch/scrubbed/${USER}/ft-ppi/artifacts/helpsteer2_lora_worker_quickdiag}"
mkdir -p "$OUTPUT_DIR/cells"
FEATURES="${FEATURES:-delta_log_length_scale,delta_log_sentences_scale}"
TARGETS="${TARGETS:-delta_log_length_scale}"
S_GRID="${S_GRID:-50,150,300,700}"
REPLICATIONS="${REPLICATIONS:-1}"
METHODS="${METHODS:-mse_stop_mse,mse_stop_ifvar,ifvar_stop_ifvar}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
MAX_LENGTH="${MAX_LENGTH:-384}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
PATIENCE="${PATIENCE:-1}"
VALIDATION_LIMIT="${VALIDATION_LIMIT:-512}"
EVALUATION_LIMIT="${EVALUATION_LIMIT:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-$NUM_WORKERS}"
NO_4BIT="${NO_4BIT:-1}"
NO_GRADIENT_CHECKPOINTING="${NO_GRADIENT_CHECKPOINTING:-1}"

echo "prewarming_huggingface_cache model_name=$MODEL_NAME hf_home=$HF_HOME"
MODEL_NAME="$MODEL_NAME" python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_name = os.environ["MODEL_NAME"]
path = snapshot_download(
    repo_id=model_name,
    ignore_patterns=[
        "*.msgpack",
        "*.h5",
        "*.ot",
        "tf_model*",
        "flax_model*",
        "onnx/*",
    ],
)
print(f"snapshot_downloaded path={path}", flush=True)
PY
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

python -m src.experiments.helpsteer2_lora_scaling \
  --input-csv "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --features "$FEATURES" \
  --targets "$TARGETS" \
  --s-grid "$S_GRID" \
  --replications "$REPLICATIONS" \
  --methods "$METHODS" \
  make-plan

PLAN_CSV="$OUTPUT_DIR/cell_plan.csv"
N_CELLS="$(python - <<PY
import pandas as pd
print(len(pd.read_csv("$PLAN_CSV")))
PY
)"
LAST_WORKER=$((NUM_WORKERS - 1))
echo "plan_csv=$PLAN_CSV n_cells=$N_CELLS num_workers=$NUM_WORKERS last_worker=$LAST_WORKER"

export HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-$NUM_WORKERS}"
GPU_ARGS="$(bash scripts/choose_hyak_gpu.sh)"
echo "gpu_args=$GPU_ARGS array_concurrency=$ARRAY_CONCURRENCY model_name=$MODEL_NAME max_length=$MAX_LENGTH train_batch_size=$TRAIN_BATCH_SIZE eval_batch_size=$EVAL_BATCH_SIZE max_epochs=$MAX_EPOCHS patience=$PATIENCE no_4bit=$NO_4BIT no_gradient_checkpointing=$NO_GRADIENT_CHECKPOINTING validation_limit=$VALIDATION_LIMIT evaluation_limit=$EVALUATION_LIMIT"

SBATCH_CMD=(
  sbatch
  $GPU_ARGS
  --array=0-"$LAST_WORKER"%"$ARRAY_CONCURRENCY"
  --export=ALL,HYAK_VENV_DIR="$VENV_REAL",INPUT_CSV="$INPUT_CSV",OUTPUT_DIR="$OUTPUT_DIR",PLAN_CSV="$PLAN_CSV",FEATURES="$FEATURES",MODEL_NAME="$MODEL_NAME",MAX_LENGTH="$MAX_LENGTH",TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE",EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE",GRAD_ACCUM="$GRAD_ACCUM",MAX_EPOCHS="$MAX_EPOCHS",PATIENCE="$PATIENCE",VALIDATION_LIMIT="$VALIDATION_LIMIT",EVALUATION_LIMIT="$EVALUATION_LIMIT",NUM_WORKERS="$NUM_WORKERS",NO_4BIT="$NO_4BIT",NO_GRADIENT_CHECKPOINTING="$NO_GRADIENT_CHECKPOINTING",TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE",HF_HUB_OFFLINE="$HF_HUB_OFFLINE"
  slurm/run_helpsteer2_lora_worker.sbatch
)
echo "submitting: ${SBATCH_CMD[*]}"
SUBMIT_OUTPUT="$("${SBATCH_CMD[@]}")"
echo "$SUBMIT_OUTPUT"
JOB_ID="$(echo "$SUBMIT_OUTPUT" | awk '{print $NF}')"
echo "job_id=$JOB_ID"

while squeue -j "$JOB_ID" -h | grep -q .; do
  echo "squeue $(date)"
  squeue -j "$JOB_ID" || true
  completed="$(find "$OUTPUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | wc -l | tr -d ' ')"
  echo "completed_cells=$completed/$N_CELLS"
  sleep 60
done

echo "sacct final"
sacct -j "$JOB_ID" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS -P || true

echo "slurm log tails"
for log in $(ls -1t /gscratch/scrubbed/"$USER"/ft-ppi/logs/hs2-worker-"${JOB_ID}"_*.out 2>/dev/null | head -n 8); do
  echo "---- $log ----"
  tail -n 100 "$log" || true
done

COMPLETED_CELLS="$(find "$OUTPUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | wc -l | tr -d ' ')"
echo "completed_cells_final=$COMPLETED_CELLS/$N_CELLS"
if [ "$COMPLETED_CELLS" -lt "$N_CELLS" ]; then
  echo "missing cells; not aggregating" >&2
  exit 1
fi

python -m src.experiments.helpsteer2_lora_scaling \
  --input-csv "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --features "$FEATURES" \
  aggregate

echo "artifact summary"
find "$OUTPUT_DIR" -maxdepth 3 -type f \( -name '*.csv' -o -name '*.md' -o -name '*.png' -o -name '*.pdf' -o -name '*.json' \) -printf "%s %p\n" | sort -nr | head -n 80
echo "report preview"
sed -n '1,180p' "$OUTPUT_DIR/lora_scaling_report.md"

echo "helpsteer2_lora_worker_quickdiag_task_done $(date)"
