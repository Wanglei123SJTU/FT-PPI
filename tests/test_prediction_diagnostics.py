from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.prediction_diagnostics import build_prediction_diagnostics


def test_build_prediction_diagnostics_groups_by_loss_train_size_and_role(tmp_path):
    pred_dir = tmp_path / "r000_s0050_v0100" / "mse"
    pred_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4],
            "split_role": ["train", "validation", "correction", "unlabeled"],
            "y_true": [88.0, 90.0, 92.0, 94.0],
            "pred_mean": [87.0, 91.0, 91.0, 95.0],
        }
    ).to_parquet(pred_dir / "predictions.parquet")

    per_run, summary = build_prediction_diagnostics(tmp_path)
    assert set(per_run["role"]) == {"train", "validation", "correction", "unlabeled", "population"}
    assert set(summary["role"]) == {"train", "validation", "correction", "unlabeled", "population"}

    pop = summary[(summary["loss"] == "mse") & (summary["train_size"] == 50) & (summary["role"] == "population")].iloc[0]
    assert pop["n_replications"] == 1
    assert pop["mean_rmse"] == pytest.approx(1.0)
    assert pop["mean_pred_std"] > 0
