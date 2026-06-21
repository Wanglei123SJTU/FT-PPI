#!/bin/bash
set -euo pipefail

echo "helpsteer2_preference_ols_screen_task_start $(date)"

cd ~/FT-PPI
git status --short --branch

PYTHON_BIN=".venv-hyak/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

OUT_DIR="artifacts/helpsteer2_preference/regression_screen"
PAIR_CSV="Data/helpsteer2_preference_pairs.csv"

echo "python=$($PYTHON_BIN --version)"
echo "prepare HelpSteer2 preference pairs"
$PYTHON_BIN -m src.data.prepare_helpsteer2_preference \
  --output-csv "$PAIR_CSV"

echo "run OLS regression screen"
$PYTHON_BIN -m src.experiments.helpsteer2_preference_regression \
  --input-csv "$PAIR_CSV" \
  --output-dir "$OUT_DIR"

echo "prepared_pair_rows"
$PYTHON_BIN - <<'PY'
import pandas as pd
df = pd.read_csv("Data/helpsteer2_preference_pairs.csv")
print({"rows": len(df), "columns": list(df.columns)})
print(df[["split", "y_preference_strength", "delta_log_length", "delta_prompt_coverage", "delta_format"]].describe(include="all").to_string())
PY

echo "ranked_targets"
cat "$OUT_DIR/ranked_targets.csv"

echo "screening_report"
sed -n '1,220p' "$OUT_DIR/screening_report.md"

echo "helpsteer2_preference_ols_screen_task_done $(date)"
