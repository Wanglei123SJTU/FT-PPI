#!/bin/bash
set -euo pipefail

echo "helpsteer2_lora_large_s_local_snapshot_task_start $(date)"

cd "${HYAK_RUNNER_REPO_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

OLD_JOB_ID="36271307"
echo "cancel_old_job=$OLD_JOB_ID"
scancel "$OLD_JOB_ID" 2>/dev/null || true

CACHE_ROOT="/gscratch/scrubbed/${USER}/ft-ppi/cache"
mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/hf_datasets" "$CACHE_ROOT/torch" "$CACHE_ROOT/xdg"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf_datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

RUN_NAME="helpsteer2_lora_large_s_diagnostic_146"
OUTPUT_DIR="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/${RUN_NAME}"
INPUT_CSV="$OUTPUT_DIR/helpsteer2_preference_pairs.csv"
SOURCE_CSV="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/helpsteer2_lora_large_s_diagnostic_143/helpsteer2_preference_pairs.csv"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
FEATURES="delta_log_length_scale,delta_log_sentences_scale"
TARGETS="delta_log_length_scale,delta_log_sentences_scale"
S_GRID="${S_GRID:-100,300,700,1000,1500,2000}"
REPLICATIONS="${REPLICATIONS:-2}"
METHODS="${METHODS:-mse_stop_mse,ifmse_stop_ifvar,ifvar_stop_ifvar}"
MAX_LENGTH="${MAX_LENGTH:-256}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
PATIENCE="${PATIENCE:-1}"
VALIDATION_LIMIT="${VALIDATION_LIMIT:-0}"
EVALUATION_LIMIT="${EVALUATION_LIMIT:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-$NUM_WORKERS}"
NO_4BIT="${NO_4BIT:-1}"
NO_GRADIENT_CHECKPOINTING="${NO_GRADIENT_CHECKPOINTING:-1}"
SKIP_TORCH_PREFLIGHT="${SKIP_TORCH_PREFLIGHT:-1}"

VENV_REAL="${HYAK_VENV_DIR:-$PWD/.venv-hyak}"
PYTHON_BIN="$VENV_REAL/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

mkdir -p "$OUTPUT_DIR" /gscratch/scrubbed/"$USER"/ft-ppi/logs
if [ -s "$SOURCE_CSV" ]; then
  cp "$SOURCE_CSV" "$INPUT_CSV"
elif [ -s "Data/helpsteer2_preference_pairs.csv" ]; then
  cp "Data/helpsteer2_preference_pairs.csv" "$INPUT_CSV"
else
  echo "missing HelpSteer2 csv; expected $SOURCE_CSV or Data/helpsteer2_preference_pairs.csv" >&2
  exit 1
fi

"$PYTHON_BIN" - <<PY
import pandas as pd
path = "$INPUT_CSV"
df = pd.read_csv(path)
print(f"input_rows={len(df)} path={path}")
print(df[["y_preference_strength","delta_log_length_scale","delta_log_sentences_scale"]].describe().to_string())
PY

echo "prewarming_huggingface_cache base_model_name=$BASE_MODEL_NAME hf_home=$HF_HOME"
SNAPSHOT_PATH="$(BASE_MODEL_NAME="$BASE_MODEL_NAME" "$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["BASE_MODEL_NAME"],
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "tf_model*", "flax_model*", "onnx/*"],
)
print(path)
PY
)"
echo "snapshot_path=$SNAPSHOT_PATH"
if [ ! -s "$SNAPSHOT_PATH/config.json" ]; then
  echo "snapshot missing config.json: $SNAPSHOT_PATH" >&2
  find "$SNAPSHOT_PATH" -maxdepth 1 -type f -print >&2 || true
  exit 1
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
MODEL_NAME="$SNAPSHOT_PATH"

"$PYTHON_BIN" -m src.experiments.helpsteer2_lora_scaling \
  --input-csv "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --features "$FEATURES" \
  --targets "$TARGETS" \
  --s-grid "$S_GRID" \
  --replications "$REPLICATIONS" \
  --methods "$METHODS" \
  make-plan

PLAN_CSV="$OUTPUT_DIR/cell_plan.csv"
N_CELLS="$("$PYTHON_BIN" - <<PY
import pandas as pd
print(len(pd.read_csv("$PLAN_CSV")))
PY
)"
LAST_WORKER=$((NUM_WORKERS - 1))
echo "plan_csv=$PLAN_CSV n_cells=$N_CELLS output_dir=$OUTPUT_DIR"

GPU_ARGS="$(HYAK_GPU_MIN_IDLE="$NUM_WORKERS" bash scripts/choose_hyak_gpu.sh)"
echo "selected_gpu_args=$GPU_ARGS array_concurrency=$ARRAY_CONCURRENCY"

SBATCH_CMD=(
  sbatch
  $GPU_ARGS
  --time=04:00:00
  --array=0-"$LAST_WORKER"%"$ARRAY_CONCURRENCY"
  --export=ALL,HYAK_VENV_DIR="$VENV_REAL",INPUT_CSV="$INPUT_CSV",OUTPUT_DIR="$OUTPUT_DIR",PLAN_CSV="$PLAN_CSV",FEATURES="$FEATURES",MODEL_NAME="$MODEL_NAME",MAX_LENGTH="$MAX_LENGTH",TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE",EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE",GRAD_ACCUM="$GRAD_ACCUM",MAX_EPOCHS="$MAX_EPOCHS",PATIENCE="$PATIENCE",VALIDATION_LIMIT="$VALIDATION_LIMIT",EVALUATION_LIMIT="$EVALUATION_LIMIT",NUM_WORKERS="$NUM_WORKERS",NO_4BIT="$NO_4BIT",NO_GRADIENT_CHECKPOINTING="$NO_GRADIENT_CHECKPOINTING",TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE",HF_HUB_OFFLINE="$HF_HUB_OFFLINE",HF_HOME="$HF_HOME",HF_DATASETS_CACHE="$HF_DATASETS_CACHE",TORCH_HOME="$TORCH_HOME",XDG_CACHE_HOME="$XDG_CACHE_HOME",TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE",SKIP_TORCH_PREFLIGHT="$SKIP_TORCH_PREFLIGHT",OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1
  slurm/run_helpsteer2_lora_worker.sbatch
)
echo "submitting large-s local snapshot diagnostic: ${SBATCH_CMD[*]}"
SUBMIT_OUTPUT="$("${SBATCH_CMD[@]}")"
echo "$SUBMIT_OUTPUT"
JOB_ID="$(echo "$SUBMIT_OUTPUT" | awk '{print $NF}')"
echo "large_s_local_snapshot_job_id=$JOB_ID"

start_ts="$(date +%s)"
while [ -n "$(squeue -j "$JOB_ID" -h || true)" ]; do
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  echo "squeue $(date) elapsed=${elapsed}s"
  squeue -j "$JOB_ID" || true
  completed="$(find "$OUTPUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | wc -l | tr -d ' ')"
  echo "completed_cells=$completed/$N_CELLS"
  mapfile -t logs < <(find /gscratch/scrubbed/"$USER"/ft-ppi/logs -maxdepth 1 -name "hs2-worker-${JOB_ID}_*.out" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 4 | cut -d' ' -f2-)
  for log in "${logs[@]}"; do
    echo "---- live tail $log ----"
    tail -n 80 "$log" || true
  done
  sleep 60
done

echo "sacct final"
sacct -j "$JOB_ID" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS -P || true

COMPLETED_CELLS="$(find "$OUTPUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | wc -l | tr -d ' ')"
echo "completed_cells_final=$COMPLETED_CELLS/$N_CELLS"
if [ "$COMPLETED_CELLS" -lt "$N_CELLS" ]; then
  echo "missing cells; printing worker log tails" >&2
  for log in /gscratch/scrubbed/"$USER"/ft-ppi/logs/hs2-worker-"${JOB_ID}"_*.out; do
    [ -e "$log" ] || continue
    echo "---- final tail $log ----"
    tail -n 160 "$log" || true
  done
  exit 1
fi

"$PYTHON_BIN" -m src.experiments.helpsteer2_lora_scaling \
  --input-csv "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --features "$FEATURES" \
  aggregate

echo "report preview"
sed -n '1,280p' "$OUTPUT_DIR/lora_scaling_report.md"

echo "helpsteer2_lora_large_s_local_snapshot_task_done $(date)"
