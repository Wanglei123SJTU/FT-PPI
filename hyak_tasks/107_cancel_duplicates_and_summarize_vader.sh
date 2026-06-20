#!/bin/bash
set -euo pipefail

echo "107_cancel_duplicates_and_summarize_vader_task_start"
hostname
date
git rev-parse --short HEAD

ROOT="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/upworthy_question_vader_current_balanced_focused_1p5b"

echo "== CANCEL ACTIVE DUPLICATE UPWORTHY ARRAYS =="
active_jobs=$(squeue -u "$USER" -h -o "%i %j %T" | awk '$2 ~ /^upworthy/ {print $1}' | cut -d_ -f1 | sort -u || true)
if [ -z "${active_jobs}" ]; then
  echo "no_active_upworthy_jobs"
else
  echo "${active_jobs}" | while read -r job_id; do
    [ -z "$job_id" ] && continue
    echo "cancelling_upworthy_job=${job_id}"
    squeue -j "$job_id" || true
    scancel "$job_id" || true
  done
fi

echo "== FINAL VADER SUMMARY =="
.venv-hyak/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("/gscratch/scrubbed/lei0603/ft-ppi/artifacts/upworthy_question_vader_current_balanced_focused_1p5b")
print("root", root)
for name in [
    "scaling_cell_metrics.csv",
    "scaling_by_s_summary.csv",
    "break_even_diagnostics.csv",
    "scaling_fit_parameters_raw.csv",
]:
    path = root / name
    print(name, "exists", path.exists(), "size", path.stat().st_size if path.exists() else None)

cell = pd.read_csv(root / "scaling_cell_metrics.csv")
by_s = pd.read_csv(root / "scaling_by_s_summary.csv")
be = pd.read_csv(root / "break_even_diagnostics.csv")
fit = pd.read_csv(root / "scaling_fit_parameters_raw.csv")

print("\ncell_count", len(cell))
print("target_feature", by_s["target_feature"].iloc[0])
print("feature_columns", by_s["feature_columns"].iloc[0])
print("target_beta_raw", by_s["target_coefficient_raw"].iloc[0])
print("direct_ols_ifvar_target_raw", by_s["direct_ols_ifvar_target_raw"].iloc[0])

cols = [
    "method",
    "s_train",
    "n_replications",
    "mean_ifvarq_raw",
    "ratio_to_constant_ifvarq",
    "ratio_to_direct_ols_ifvarq",
    "drop_from_direct_ols_ifvarq_pct",
    "mean_rmse_scaled",
    "mean_corr",
]
print("\n-- by_s_summary --")
print(by_s[cols].sort_values(["method", "s_train"]).to_string(index=False))

print("\n-- break_even winners --")
winners = be[be["beats_labeled_only_proxy"]].copy()
if winners.empty:
    print("none")
else:
    winners = winners.sort_values(["budget_B", "var_ratio_to_labeled_only", "method", "s_train"])
    print(winners.to_string(index=False))

print("\n-- best row per budget --")
best = be.sort_values(["budget_B", "var_ratio_to_labeled_only"]).groupby("budget_B", as_index=False).head(1)
print(best.to_string(index=False))

print("\n-- scaling fit --")
print(fit.to_string(index=False))
PY

echo "107_cancel_duplicates_and_summarize_vader_task_done"
