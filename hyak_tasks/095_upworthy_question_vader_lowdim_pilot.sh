#!/bin/bash
set -euo pipefail

echo "upworthy_question_vader_lowdim_pilot_task_start"
hostname
date
git rev-parse --short HEAD

export CONFIG="configs/upworthy_question_vader_textonly_lowdim_pilot_1p5b.yaml"
export ARRAY_CONC="${ARRAY_CONC:-8}"
export HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-8}"
unset HYAK_FORCE_GPU_ARGS

echo "== CHECK UPGRADED UPWORTHY DATA =="
if ! .venv-hyak/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("Data/upworthy_pairs_with_text_features.csv")
required = {"delta_QUESTION", "delta_VADER_COMPOUND"}
if not path.exists():
    raise SystemExit("missing Data/upworthy_pairs_with_text_features.csv")
cols = set(pd.read_csv(path, nrows=1).columns)
missing = sorted(required - cols)
if missing:
    raise SystemExit("current Data/upworthy_pairs_with_text_features.csv is missing required columns: " + ", ".join(missing))
print("data_ok columns_present=" + ",".join(sorted(required)))
PY
then
  echo "== UPGRADE DATA FROM PAIR HEADLINES =="
  UPGRADE_DIR="artifacts/upworthy_m_estimation/hyak_data_upgrade"
  UPGRADED_CSV="$UPGRADE_DIR/upworthy_pairs_with_text_features.csv"
  mkdir -p "$UPGRADE_DIR"
  .venv-hyak/bin/python -m src.data.upworthy_text_features \
    --pairs Data/upworthy_pairs_with_text_features.csv \
    --from-pair-headlines \
    --output-dir "$UPGRADE_DIR" \
    --output-csv "$UPGRADED_CSV"
  cp "$UPGRADED_CSV" Data/upworthy_pairs_with_text_features.csv
  .venv-hyak/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("Data/upworthy_pairs_with_text_features.csv")
required = {"delta_QUESTION", "delta_VADER_COMPOUND"}
cols = set(pd.read_csv(path, nrows=1).columns)
missing = sorted(required - cols)
if missing:
    raise SystemExit("upgraded data still missing required columns: " + ", ".join(missing))
print("upgraded_data_ok columns_present=" + ",".join(sorted(required)))
PY
fi

bash hyak_tasks/080_upworthy_question_scaling_diagnostic_full.sh

echo "upworthy_question_vader_lowdim_pilot_task_done"
