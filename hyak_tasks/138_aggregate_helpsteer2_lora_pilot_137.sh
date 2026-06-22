#!/bin/bash
set -euo pipefail

echo "aggregate_helpsteer2_lora_pilot_137_task_start $(date)"

cd ~/FT-PPI

OUTPUT_DIR="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/helpsteer2_lora_scaling_pilot_x8_137"
INPUT_CSV="${OUTPUT_DIR}/helpsteer2_preference_pairs.csv"
FEATURES="delta_log_length_scale,delta_log_sentences_scale"
N_EXPECTED=63

if [ -d ".venv-hyak" ]; then
  # shellcheck disable=SC1091
  source .venv-hyak/bin/activate
fi

echo "repo_status"
git status --short --branch
echo "output_dir=${OUTPUT_DIR}"

if [ ! -f "$INPUT_CSV" ]; then
  echo "missing input csv: $INPUT_CSV" >&2
  exit 1
fi

completed="$(find "$OUTPUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | wc -l | tr -d ' ')"
echo "completed_cells=${completed}/${N_EXPECTED}"
if [ "$completed" -lt "$N_EXPECTED" ]; then
  echo "not enough cells to aggregate" >&2
  exit 1
fi

python -m src.experiments.helpsteer2_lora_scaling \
  --input-csv "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --features "$FEATURES" \
  aggregate

echo "artifact summary"
find "$OUTPUT_DIR" -maxdepth 3 -type f \( -name '*.csv' -o -name '*.md' -o -name '*.png' -o -name '*.json' \) -printf "%s %p\n" | sort -nr | head -n 100

echo "report preview"
sed -n '1,220p' "$OUTPUT_DIR/lora_scaling_report.md"

echo "scaling_fit_preview"
python - <<PY
import pandas as pd
from pathlib import Path
out = Path("$OUTPUT_DIR")
for name in ["lora_scaling_summary.csv", "lora_scaling_fit.csv"]:
    path = out / name
    if path.exists():
        print(f"--- {name} ---")
        print(pd.read_csv(path).to_string(index=False))
PY

echo "aggregate_helpsteer2_lora_pilot_137_task_done $(date)"
