#!/bin/bash
set -euo pipefail

echo "aggregate_helpsteer2_lora_pilot_137_retry_task_start $(date)"

cd ~/FT-PPI

OUTPUT_DIR="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/helpsteer2_lora_scaling_pilot_x8_137"
INPUT_CSV="${OUTPUT_DIR}/helpsteer2_preference_pairs.csv"
FEATURES="delta_log_length_scale,delta_log_sentences_scale"
N_EXPECTED=63
VENV_REAL="${HYAK_VENV_DIR:-$PWD/.venv-hyak}"
PYTHON_BIN="$VENV_REAL/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

echo "repo_status"
git status --short --branch
echo "python_bin=${PYTHON_BIN}"
echo "output_dir=${OUTPUT_DIR}"

completed="$(find "$OUTPUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | wc -l | tr -d ' ')"
echo "completed_cells=${completed}/${N_EXPECTED}"
if [ "$completed" -lt "$N_EXPECTED" ]; then
  echo "not enough cells to aggregate" >&2
  exit 1
fi

"$PYTHON_BIN" -m src.experiments.helpsteer2_lora_scaling \
  --input-csv "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --features "$FEATURES" \
  aggregate

echo "report preview"
sed -n '1,180p' "$OUTPUT_DIR/lora_scaling_report.md"

echo "aggregate_helpsteer2_lora_pilot_137_retry_task_done $(date)"
