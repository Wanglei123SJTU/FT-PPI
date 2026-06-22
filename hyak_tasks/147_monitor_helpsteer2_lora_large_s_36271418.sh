#!/bin/bash
set -euo pipefail

echo "monitor_helpsteer2_lora_large_s_36271418_task_start $(date)"

cd "${HYAK_RUNNER_REPO_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

JOB_ID="36271418"
OUTPUT_DIR="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/helpsteer2_lora_large_s_diagnostic_146"
INPUT_CSV="$OUTPUT_DIR/helpsteer2_preference_pairs.csv"
PLAN_CSV="$OUTPUT_DIR/cell_plan.csv"
CACHE_ROOT="/gscratch/scrubbed/${USER}/ft-ppi/cache"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf_datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface"

VENV_REAL="${HYAK_VENV_DIR:-$PWD/.venv-hyak}"
PYTHON_BIN="$VENV_REAL/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

mkdir -p "$OUTPUT_DIR/cells"

N_CELLS="$("$PYTHON_BIN" - <<PY
import pandas as pd
print(len(pd.read_csv("$PLAN_CSV")))
PY
)"
echo "job_id=$JOB_ID output_dir=$OUTPUT_DIR n_cells=$N_CELLS"

start_ts="$(date +%s)"
while [ -n "$(squeue -j "$JOB_ID" -h || true)" ]; do
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  completed="$(find "$OUTPUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | wc -l | tr -d ' ' || true)"
  echo "poll $(date) elapsed=${elapsed}s completed_cells=${completed}/${N_CELLS}"
  squeue -j "$JOB_ID" || true
  sleep 120
done

echo "sacct final"
sacct -j "$JOB_ID" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS -P || true

COMPLETED_CELLS="$(find "$OUTPUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | wc -l | tr -d ' ' || true)"
echo "completed_cells_final=$COMPLETED_CELLS/$N_CELLS"
if [ "$COMPLETED_CELLS" -lt "$N_CELLS" ]; then
  echo "missing cells; worker log tails:" >&2
  for log in /gscratch/scrubbed/"$USER"/ft-ppi/logs/hs2-worker-"${JOB_ID}"_*.out; do
    [ -e "$log" ] || continue
    echo "---- final tail $log ----"
    tail -n 200 "$log" || true
  done
  exit 1
fi

"$PYTHON_BIN" -m src.experiments.helpsteer2_lora_scaling \
  --input-csv "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --features "delta_log_length_scale,delta_log_sentences_scale" \
  aggregate

echo "report preview"
sed -n '1,280p' "$OUTPUT_DIR/lora_scaling_report.md"

echo "monitor_helpsteer2_lora_large_s_36271418_task_done $(date)"
