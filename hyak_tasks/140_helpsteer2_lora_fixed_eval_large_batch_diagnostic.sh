#!/bin/bash
set -euo pipefail

echo "helpsteer2_lora_fixed_eval_large_batch_diagnostic_task_start $(date)"

cd ~/FT-PPI

RUN_NAME="helpsteer2_lora_fixed_eval_large_batch_140"
OUTPUT_DIR="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/${RUN_NAME}"
INPUT_CSV="${OUTPUT_DIR}/helpsteer2_preference_pairs.csv"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
FEATURES="delta_log_length_scale,delta_log_sentences_scale"
TARGETS="delta_log_length_scale,delta_log_sentences_scale"
S_GRID="50,100,200,300,500,700"
REPLICATIONS="${REPLICATIONS:-2}"
METHODS="mse_stop_mse,ifmse_stop_ifvar,ifvar_stop_ifvar"
MAX_LENGTH="${MAX_LENGTH:-256}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-2}"
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
echo "python_bin=$PYTHON_BIN"

mkdir -p "$OUTPUT_DIR" /gscratch/scrubbed/"$USER"/ft-ppi/logs

if [ ! -f "$INPUT_CSV" ]; then
  echo "preparing input csv at $INPUT_CSV"
  "$PYTHON_BIN" -m src.data.prepare_helpsteer2_preference \
    --output-csv "$INPUT_CSV" \
    --limit 0
fi

echo "prewarming_huggingface_cache model_name=$MODEL_NAME"
MODEL_NAME="$MODEL_NAME" timeout 600 "$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["MODEL_NAME"],
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "tf_model*", "flax_model*", "onnx/*"],
)
print(f"snapshot_downloaded path={path}", flush=True)
PY
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

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
  --time=03:00:00
  --array=0-"$LAST_WORKER"%"$ARRAY_CONCURRENCY"
  --export=ALL,HYAK_VENV_DIR="$VENV_REAL",INPUT_CSV="$INPUT_CSV",OUTPUT_DIR="$OUTPUT_DIR",PLAN_CSV="$PLAN_CSV",FEATURES="$FEATURES",MODEL_NAME="$MODEL_NAME",MAX_LENGTH="$MAX_LENGTH",TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE",EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE",GRAD_ACCUM="$GRAD_ACCUM",MAX_EPOCHS="$MAX_EPOCHS",PATIENCE="$PATIENCE",VALIDATION_LIMIT="$VALIDATION_LIMIT",EVALUATION_LIMIT="$EVALUATION_LIMIT",NUM_WORKERS="$NUM_WORKERS",NO_4BIT="$NO_4BIT",NO_GRADIENT_CHECKPOINTING="$NO_GRADIENT_CHECKPOINTING",TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE",HF_HUB_OFFLINE="$HF_HUB_OFFLINE",SKIP_TORCH_PREFLIGHT="$SKIP_TORCH_PREFLIGHT",OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1
  slurm/run_helpsteer2_lora_worker.sbatch
)
echo "submitting fixed-eval diagnostic: ${SBATCH_CMD[*]}"
SUBMIT_OUTPUT="$("${SBATCH_CMD[@]}")"
echo "$SUBMIT_OUTPUT"
JOB_ID="$(echo "$SUBMIT_OUTPUT" | awk '{print $NF}')"
echo "fixed_eval_diagnostic_job_id=$JOB_ID"

start_ts="$(date +%s)"
while squeue -j "$JOB_ID" -h | grep -q .; do
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  echo "squeue $(date) elapsed=${elapsed}s"
  squeue -j "$JOB_ID" || true
  completed="$(find "$OUTPUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | wc -l | tr -d ' ')"
  echo "completed_cells=$completed/$N_CELLS"
  for log in $(ls -1t /gscratch/scrubbed/"$USER"/ft-ppi/logs/hs2-worker-"${JOB_ID}"_*.out 2>/dev/null | head -n 4); do
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
  echo "missing cells; not aggregating full diagnostic" >&2
  exit 1
fi

"$PYTHON_BIN" -m src.experiments.helpsteer2_lora_scaling \
  --input-csv "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --features "$FEATURES" \
  aggregate

echo "report preview"
sed -n '1,220p' "$OUTPUT_DIR/lora_scaling_report.md"

echo "helpsteer2_lora_fixed_eval_large_batch_diagnostic_task_done $(date)"
