#!/bin/bash
set -euo pipefail

TASK_ID="116_run_question_vaderpos_common_confirm"
LOCK_DIR=".hyak_runner/task_locks"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/${TASK_ID}.lock"
if ! flock -n 9; then
  echo "${TASK_ID}_already_running_elsewhere"
  exit 0
fi

echo "${TASK_ID}_task_start"
hostname
date
git rev-parse --short HEAD

run_config() {
  local config="$1"
  local label="$2"
  local root="$3"

  echo "== DESCRIBE ${label} CONFIG =="
  .venv-hyak/bin/python -m src.experiments.upworthy_question_scaling_law describe --config "$config"

  echo "== RUN ${label} CONFIRMATION =="
  export CONFIG="$config"
  export ARRAY_CONC="${ARRAY_CONC:-8}"
  unset HYAK_FORCE_GPU_ARGS
  HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-$ARRAY_CONC}" bash hyak_tasks/080_upworthy_question_scaling_diagnostic_full.sh

  echo "== ${label} SUMMARY =="
  .venv-hyak/bin/python - "$root" <<'PY'
from pathlib import Path
import pandas as pd
import sys

root = Path(sys.argv[1])
by_s = pd.read_csv(root / "scaling_by_s_summary.csv")
be = pd.read_csv(root / "break_even_diagnostics.csv")
fit = pd.read_csv(root / "scaling_fit_parameters_raw.csv")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)

print("root", root)
print("target_feature", by_s["target_feature"].iloc[0])
print("feature_columns", by_s["feature_columns"].iloc[0])
print("target_beta_raw", by_s["target_coefficient_raw"].iloc[0])
print("direct_ols_ifvar_target_raw", by_s["direct_ols_ifvar_target_raw"].iloc[0])

cols = [
    "method",
    "s_train",
    "n_replications",
    "mean_ifvarq_raw",
    "se_ifvarq_raw",
    "ratio_to_direct_ols_ifvarq",
    "drop_from_direct_ols_ifvarq_pct",
    "mean_corr",
]
print("\n-- by_s_summary --")
print(by_s[cols].sort_values(["method", "s_train"]).to_string(index=False))

print("\n-- best row per budget --")
best = be.sort_values(["budget_B", "var_ratio_to_labeled_only"]).groupby("budget_B", as_index=False).head(1)
print(best.to_string(index=False))

print("\n-- IFVar budget winners --")
ifvar_winners = be[
    be["method"].str.contains("ifvar", case=False, na=False)
    & be["beats_labeled_only_proxy"]
].copy()
if ifvar_winners.empty:
    print("none")
else:
    ifvar_winners = ifvar_winners.sort_values(["budget_B", "var_ratio_to_labeled_only", "method", "s_train"])
    print(ifvar_winners.to_string(index=False))

print("\n-- all budget winners --")
winners = be[be["beats_labeled_only_proxy"]].copy()
if winners.empty:
    print("none")
else:
    winners = winners.sort_values(["budget_B", "var_ratio_to_labeled_only", "method", "s_train"])
    print(winners.to_string(index=False))

print("\n-- scaling fit --")
print(fit.sort_values("r2", ascending=False).to_string(index=False))
PY
}

run_config \
  "configs/upworthy_question_vaderpos_confirm_10rep_1p5b.yaml" \
  "QUESTION_VADER_POS" \
  "/gscratch/scrubbed/${USER}/ft-ppi/artifacts/upworthy_question_vaderpos_confirm_10rep_1p5b"

run_config \
  "configs/upworthy_question_common_confirm_10rep_1p5b.yaml" \
  "QUESTION_COMMON" \
  "/gscratch/scrubbed/${USER}/ft-ppi/artifacts/upworthy_question_common_confirm_10rep_1p5b"

echo "${TASK_ID}_task_done"
