from __future__ import annotations

import pandas as pd

from src.experiments.run_allocation_diagnostic import add_diagnostic_columns, build_allocation_runs


def test_build_allocation_runs_sets_train_sizes_and_output_dirs():
    config = {
        "output_dir": "artifacts/allocation_diagnostic",
        "budget": 1000,
        "population_size": 10000,
        "validation_size": 100,
        "allocation_ratios": [0.05, 0.1, 0.2],
    }
    runs = build_allocation_runs(config)
    assert [run["train_size"] for run in runs] == [50, 100, 200]
    assert [run["validation_size"] for run in runs] == [100, 100, 100]
    assert [run["allocation_tag"] for run in runs] == ["s0050_v0100", "s0100_v0100", "s0200_v0100"]
    assert runs[0]["output_dir"].endswith("s0050_v0100")


def test_build_allocation_runs_supports_multiple_budgets():
    config = {
        "output_dir": "artifacts/first_pilot",
        "budgets": [500, 1000],
        "population_size": 20000,
        "validation_size": 100,
        "allocation_ratios": [0.1, 0.2],
    }
    runs = build_allocation_runs(config)
    assert [run["allocation_tag"] for run in runs] == [
        "B0500_s0050_v0100",
        "B0500_s0100_v0100",
        "B1000_s0100_v0100",
        "B1000_s0200_v0100",
    ]
    assert [run["budget"] for run in runs] == [500, 500, 1000, 1000]


def test_add_diagnostic_columns_maps_loss_and_residual_variance():
    metrics = pd.DataFrame(
        {
            "method": ["sample_mean", "lora_mse+ppi_plus", "lora_var+ppi_plus"],
            "prediction_file": [
                "artifacts/allocation_diagnostic/s0050_v0100/mse/predictions.parquet",
                "artifacts/allocation_diagnostic/s0050_v0100/mse/predictions.parquet",
                "artifacts/allocation_diagnostic/s0050_v0100/var/predictions.parquet",
            ],
            "estimated_variance": [0.1, 0.2, 0.3],
        }
    )
    run_cfg = {
        "allocation_ratio": 0.05,
        "train_size": 50,
        "validation_size": 100,
        "budget": 1000,
        "population_size": 10000,
        "replication_id": 0,
    }
    out = add_diagnostic_columns(metrics, run_cfg, {"mse": 1.5, "var": 2.5})
    assert out["loss"].tolist() == ["sample_mean", "mse", "var"]
    assert pd.isna(out.loc[0, "residual_variance"])
    assert out.loc[1, "residual_variance"] == 1.5
    assert out.loc[2, "estimated_estimator_variance"] == 0.3
