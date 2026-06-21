#!/bin/bash
set -euo pipefail

echo "upworthy_question_vader_small_s_v2_quick_fallback_task_start $(date)"
hostname
git rev-parse --short HEAD

echo "== VALIDATE QUICK FALLBACK V2 DATA =="
.venv-hyak/bin/python - <<'PY'
import pandas as pd

path = "Data/upworthy_pairs_with_text_features_v2.csv.gz"
df = pd.read_csv(path, nrows=1000)
required = ["delta_QUESTION", "delta_VADER_COMPOUND", "headline_a", "headline_b"]
missing = [col for col in required if col not in df.columns]
if missing:
    raise SystemExit(f"missing required v2 columns: {missing}")
print(
    "quick_fallback_v2_columns_ok",
    {col: float((df[col] != 0).mean()) if col.startswith("delta_") else "text" for col in required},
)
PY

export CONFIGS="configs/upworthy_question_vader_small_s_v2_quick_fallback_1p5b.yaml"
export ARRAY_CONC="${ARRAY_CONC:-4}"
export HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-4}"
echo "CONFIGS=$CONFIGS"
echo "ARRAY_CONC=$ARRAY_CONC"
echo "HYAK_GPU_MIN_IDLE=$HYAK_GPU_MIN_IDLE"

bash hyak_tasks/095_upworthy_clean_diagnostic.sh

echo "upworthy_question_vader_small_s_v2_quick_fallback_task_done $(date)"
