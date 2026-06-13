#!/bin/bash
set -euo pipefail

echo "prediction_diagnostics_task_start"
hostname
date
git rev-parse --short HEAD

.venv-hyak/bin/python - <<'PY'
from pathlib import Path

import pandas as pd

from src.analysis.prediction_diagnostics import write_prediction_diagnostics

root = Path("artifacts/allocation_scaling_probe")
summary_dir = root / "summary"
_, summary = write_prediction_diagnostics(root, summary_dir)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

cols = [
    "loss",
    "train_size",
    "role",
    "n_replications",
    "mean_corr",
    "sd_corr",
    "mean_rmse",
    "mean_bias",
    "mean_pred_mean",
    "mean_pred_std",
    "mean_y_std",
    "mean_residual_variance",
]
print("== prediction diagnostics population/correction ==")
target = summary[summary["role"].isin(["population", "correction"])].sort_values(["loss", "role", "train_size"])
print(target[cols].to_string(index=False))

curve = pd.read_csv(summary_dir / "allocation_curve.csv")
ppi = curve[curve["method"].astype(str).str.contains("ppi_plus", regex=False)].copy()
print("== ppi_plus normalized variance ==")
print(
    ppi[
        [
            "method",
            "loss",
            "train_size",
            "mean_estimated_variance",
            "normalized_estimated_variance",
            "mean_sample_savings",
        ]
    ].sort_values(["method", "train_size"]).to_string(index=False)
)
PY

echo "prediction_diagnostics_task_done"
