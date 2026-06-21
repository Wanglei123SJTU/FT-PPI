#!/bin/bash
set -euo pipefail

echo "helpsteer2_lora_length_two_feature_pilot_task_start $(date)"

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

if ! .venv-hyak/bin/python - <<'PY'
import importlib.util
import sys

mods = ["torch", "transformers", "datasets", "accelerate", "peft", "bitsandbytes"]
missing = [mod for mod in mods if importlib.util.find_spec(mod) is None]
if missing:
    print("missing_gpu_deps=" + ",".join(missing))
    sys.exit(1)
PY
then
  echo "repairing incomplete Hyak Python environment"
  FORCE_INSTALL=1 bash scripts/setup_hyak_env.sh
fi

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

OUTPUT_DIR="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/helpsteer2_lora_length_two_feature_pilot"
mkdir -p "$OUTPUT_DIR/cells"
FEATURES="delta_log_length_scale,delta_log_sentences_scale"
TARGETS="delta_log_length_scale"
S_GRID="50,100,150,200,300,400,500,700"
REPLICATIONS="3"
METHODS="mse_stop_mse,mse_stop_ifvar,ifvar_stop_ifvar"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"

python -m src.experiments.helpsteer2_lora_scaling \
  --input-csv "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --features "$FEATURES" \
  --targets "$TARGETS" \
  describe

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
LAST_INDEX=$((N_CELLS - 1))
echo "plan_csv=$PLAN_CSV n_cells=$N_CELLS last_index=$LAST_INDEX"

export HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-8}"
GPU_ARGS="$(bash scripts/choose_hyak_gpu.sh)"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-8}"
echo "gpu_args=$GPU_ARGS array_concurrency=$ARRAY_CONCURRENCY model_name=$MODEL_NAME"

SBATCH_CMD=(
  sbatch
  $GPU_ARGS
  --array=0-"$LAST_INDEX"%"$ARRAY_CONCURRENCY"
  --export=ALL,INPUT_CSV="$INPUT_CSV",OUTPUT_DIR="$OUTPUT_DIR",PLAN_CSV="$PLAN_CSV",FEATURES="$FEATURES",MODEL_NAME="$MODEL_NAME"
  slurm/run_helpsteer2_lora_scaling.sbatch
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
  sleep 120
done

echo "sacct final"
sacct -j "$JOB_ID" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS -P || true

echo "slurm log tails"
for log in $(ls -1t /gscratch/scrubbed/"$USER"/ft-ppi/logs/hs2-lora-"${JOB_ID}"_*.out 2>/dev/null | head -n 8); do
  echo "---- $log ----"
  tail -n 80 "$log" || true
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
sed -n '1,160p' "$OUTPUT_DIR/lora_scaling_report.md"

echo "helpsteer2_lora_length_two_feature_pilot_task_done $(date)"
