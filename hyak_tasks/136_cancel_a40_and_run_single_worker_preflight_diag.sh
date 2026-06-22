#!/bin/bash
set -euo pipefail

echo "cancel_a40_and_run_single_worker_preflight_diag_task_start $(date)"

cd "${HYAK_RUNNER_REPO_DIR:-$PWD}"
git status --short --branch

OLD_JOB_ID="${OLD_JOB_ID:-36268601}"
old_state="$(squeue -j "$OLD_JOB_ID" -h -o '%T' 2>/dev/null | head -n 1 || true)"
echo "old_job_id=$OLD_JOB_ID old_state=${old_state:-not_in_queue}"
if [ -n "$old_state" ]; then
  echo "canceling old worker job $OLD_JOB_ID"
  scancel "$OLD_JOB_ID" || true
fi

echo "gpu_snapshot"
sinfo -o "%20P %18G %8D %8t %10C %10m %N" | grep -Ei 'gpu|ckpt|h200|a100|a40|l40|rtx6k' || true

module purge
module load cuda/12.4.1
module load foster/python/miniconda/3.8 2>/dev/null || \
  module load chem/miniconda3/3.8 2>/dev/null || \
  module load coenv/miniconda/3.13.11 2>/dev/null || \
  module load coenv/python/3.11.9

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

INPUT_CSV="Data/helpsteer2_preference_pairs.csv"
if [ ! -s "$INPUT_CSV" ]; then
  echo "preparing HelpSteer2 preference data"
  python -m src.data.prepare_helpsteer2_preference \
    --output-csv "$INPUT_CSV" \
    --summary-json "Data/helpsteer2_preference_pairs.summary.json"
fi

OUTPUT_DIR="${OUTPUT_DIR:-/gscratch/scrubbed/${USER}/ft-ppi/artifacts/helpsteer2_lora_worker_136_single_preflight_diag}"
mkdir -p "$OUTPUT_DIR/cells"
FEATURES="${FEATURES:-delta_log_length_scale,delta_log_sentences_scale}"
TARGETS="${TARGETS:-delta_log_length_scale}"
S_GRID="${S_GRID:-50}"
REPLICATIONS="${REPLICATIONS:-1}"
METHODS="${METHODS:-mse_stop_mse,ifvar_stop_ifvar}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
MAX_LENGTH="${MAX_LENGTH:-256}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-2}"
PATIENCE="${PATIENCE:-1}"
VALIDATION_LIMIT="${VALIDATION_LIMIT:-256}"
EVALUATION_LIMIT="${EVALUATION_LIMIT:-512}"
NUM_WORKERS=1
NO_4BIT="${NO_4BIT:-1}"
NO_GRADIENT_CHECKPOINTING="${NO_GRADIENT_CHECKPOINTING:-1}"
TORCH_PREFLIGHT_TIMEOUT="${TORCH_PREFLIGHT_TIMEOUT:-180}"
EXCLUDE_NODES="${EXCLUDE_NODES:-g3060}"

echo "prewarming_huggingface_cache model_name=$MODEL_NAME hf_home=$HF_HOME"
MODEL_NAME="$MODEL_NAME" timeout 600 python - <<'PY'
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
echo "plan_csv=$PLAN_CSV n_cells=$N_CELLS num_workers=$NUM_WORKERS output_dir=$OUTPUT_DIR"

GPU_ARGS="$(HYAK_GPU_MIN_IDLE=1 bash scripts/choose_hyak_gpu.sh)"
echo "selected_gpu_args=$GPU_ARGS exclude_nodes=$EXCLUDE_NODES"

SBATCH_CMD=(
  sbatch
  $GPU_ARGS
  --exclude="$EXCLUDE_NODES"
  --time=01:00:00
  --array=0-0%1
  --export=ALL,HYAK_VENV_DIR="$VENV_REAL",INPUT_CSV="$INPUT_CSV",OUTPUT_DIR="$OUTPUT_DIR",PLAN_CSV="$PLAN_CSV",FEATURES="$FEATURES",MODEL_NAME="$MODEL_NAME",MAX_LENGTH="$MAX_LENGTH",TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE",EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE",GRAD_ACCUM="$GRAD_ACCUM",MAX_EPOCHS="$MAX_EPOCHS",PATIENCE="$PATIENCE",VALIDATION_LIMIT="$VALIDATION_LIMIT",EVALUATION_LIMIT="$EVALUATION_LIMIT",NUM_WORKERS="$NUM_WORKERS",NO_4BIT="$NO_4BIT",NO_GRADIENT_CHECKPOINTING="$NO_GRADIENT_CHECKPOINTING",TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE",HF_HUB_OFFLINE="$HF_HUB_OFFLINE",TORCH_PREFLIGHT_TIMEOUT="$TORCH_PREFLIGHT_TIMEOUT",OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1
  slurm/run_helpsteer2_lora_worker.sbatch
)
echo "submitting single-worker diagnostic: ${SBATCH_CMD[*]}"
SUBMIT_OUTPUT="$("${SBATCH_CMD[@]}")"
echo "$SUBMIT_OUTPUT"
JOB_ID="$(echo "$SUBMIT_OUTPUT" | awk '{print $NF}')"
echo "single_worker_job_id=$JOB_ID"

start_ts="$(date +%s)"
while squeue -j "$JOB_ID" -h | grep -q .; do
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  echo "squeue $(date) elapsed=${elapsed}s"
  squeue -j "$JOB_ID" || true
  completed="$(find "$OUTPUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | wc -l | tr -d ' ')"
  echo "completed_cells=$completed/$N_CELLS"
  for log in $(ls -1t /gscratch/scrubbed/"$USER"/ft-ppi/logs/hs2-worker-"${JOB_ID}"_*.out 2>/dev/null | head -n 2); do
    echo "---- live tail $log ----"
    tail -n 60 "$log" || true
  done
  if [ "$completed" -eq 0 ] && [ "$elapsed" -gt 1200 ]; then
    echo "no cells completed after ${elapsed}s; canceling $JOB_ID" >&2
    scancel "$JOB_ID" || true
    break
  fi
  sleep 30
done

echo "sacct final"
sacct -j "$JOB_ID" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS -P || true

echo "slurm log tails"
for log in $(ls -1t /gscratch/scrubbed/"$USER"/ft-ppi/logs/hs2-worker-"${JOB_ID}"_*.out 2>/dev/null | head -n 8); do
  echo "---- $log ----"
  tail -n 180 "$log" || true
done

COMPLETED_CELLS="$(find "$OUTPUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | wc -l | tr -d ' ')"
echo "completed_cells_final=$COMPLETED_CELLS/$N_CELLS"
if [ "$COMPLETED_CELLS" -lt "$N_CELLS" ]; then
  echo "missing cells; diagnostic failed before aggregation" >&2
  exit 1
fi

python -m src.experiments.helpsteer2_lora_scaling \
  --input-csv "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --features "$FEATURES" \
  aggregate

echo "artifact summary"
find "$OUTPUT_DIR" -maxdepth 3 -type f \( -name '*.csv' -o -name '*.md' -o -name '*.png' -o -name '*.json' \) -printf "%s %p\n" | sort -nr | head -n 80
echo "report preview"
sed -n '1,180p' "$OUTPUT_DIR/lora_scaling_report.md"

echo "cancel_a40_and_run_single_worker_preflight_diag_task_done $(date)"
