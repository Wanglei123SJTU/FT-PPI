#!/bin/bash
set -euo pipefail

echo "print_scaling_probe_summary_task_start"
hostname
date
git rev-parse --short HEAD

.venv-hyak/bin/python - <<'PY'
from pathlib import Path

import pandas as pd

root = Path("artifacts/allocation_scaling_probe/summary")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

for name in [
    "oracle_allocations.csv",
    "scaling_law_fit.csv",
    "scaling_law_loso.csv",
]:
    path = root / name
    print(f"== {name} ==")
    if not path.exists():
        print(f"missing {path}")
        continue
    print(pd.read_csv(path).to_string(index=False))

curve_path = root / "allocation_curve.csv"
print("== allocation_curve ppi_plus ==")
curve = pd.read_csv(curve_path)
cols = [
    "method",
    "loss",
    "train_size",
    "allocation_ratio",
    "n_replications",
    "mean_residual_variance",
    "mean_estimated_variance",
    "normalized_estimated_variance",
    "mean_ci_length",
    "mean_sample_savings",
]
target = curve[curve["method"].astype(str).str.contains("ppi_plus", regex=False)].copy()
target = target.sort_values(["method", "loss", "train_size"])
print(target[cols].to_string(index=False))

metrics = pd.read_csv(root / "allocation_metrics.csv")
print("== row counts ==")
print(metrics.groupby(["replication_id", "train_size"]).size().unstack(fill_value=0).to_string())
PY

echo "print_scaling_probe_summary_task_done"
