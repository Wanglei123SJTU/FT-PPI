#!/bin/bash
set -euo pipefail

echo "monitor_helpsteer2_lora_large_s_36271307_task_start $(date)"

cd "${HYAK_RUNNER_REPO_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

CACHE_ROOT="/gscratch/scrubbed/${USER}/ft-ppi/cache"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf_datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

JOB_ID="36271307"
OUTPUT_DIR="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/helpsteer2_lora_large_s_diagnostic_143"
INPUT_CSV="$OUTPUT_DIR/helpsteer2_preference_pairs.csv"
PLAN_CSV="$OUTPUT_DIR/cell_plan.csv"
VENV_REAL="${HYAK_VENV_DIR:-$PWD/.venv-hyak}"
PYTHON_BIN="$VENV_REAL/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

echo "job_id=$JOB_ID"
echo "output_dir=$OUTPUT_DIR"
echo "python_bin=$PYTHON_BIN"

if [ ! -s "$PLAN_CSV" ]; then
  echo "missing plan csv: $PLAN_CSV" >&2
  exit 1
fi

N_CELLS="$("$PYTHON_BIN" - <<PY
import pandas as pd
print(len(pd.read_csv("$PLAN_CSV")))
PY
)"
echo "n_cells=$N_CELLS"

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
  --features "delta_log_length_scale,delta_log_sentences_scale" \
  aggregate

echo "report preview"
sed -n '1,260p' "$OUTPUT_DIR/lora_scaling_report.md"

echo "monitor_helpsteer2_lora_large_s_36271307_task_done $(date)"
