from __future__ import annotations

import pandas as pd
import pytest

from src.experiments.run_allocation_diagnostic import (
    _completed_run_metrics,
    add_diagnostic_columns,
    build_allocation_runs,
    summarize_allocation_curve,
    summarize_oracle_allocations,
    summarize_scaling_laws,
)


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


def test_build_allocation_runs_supports_replications_and_split_seed():
    config = {
        "output_dir": "artifacts/allocation_scaling_probe",
        "seed": 1,
        "population_seed": 10,
        "split_seed": 20,
        "replication_ids": [0, 1, 2],
        "budget": 1000,
        "population_size": 10000,
        "validation_size": 100,
        "allocation_ratios": [0.05, 0.075],
    }
    runs = build_allocation_runs(config)
    assert [run["allocation_tag"] for run in runs] == [
        "r000_s0050_v0100",
        "r000_s0075_v0100",
        "r001_s0050_v0100",
        "r001_s0075_v0100",
        "r002_s0050_v0100",
        "r002_s0075_v0100",
    ]
    assert {run["population_seed"] for run in runs} == {10}
    assert [run["split_seed"] for run in runs] == [20, 20, 21, 21, 22, 22]


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


def test_completed_run_metrics_requires_metrics_and_predictions(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "summary").mkdir(parents=True)
    (run_dir / "mse").mkdir()
    (run_dir / "var").mkdir()
    pd.DataFrame({"method": ["sample_mean"]}).to_csv(run_dir / "summary" / "metrics.csv", index=False)
    assert _completed_run_metrics(run_dir, ["mse", "var"]) is None

    pd.DataFrame({"x": [1]}).to_parquet(run_dir / "mse" / "predictions.parquet")
    pd.DataFrame({"x": [1]}).to_parquet(run_dir / "var" / "predictions.parquet")
    completed = _completed_run_metrics(run_dir, ["mse", "var"])
    assert completed is not None
    assert completed["method"].tolist() == ["sample_mean"]


def test_allocation_summary_tables_average_replications_and_find_oracle():
    metrics = pd.DataFrame(
        {
            "budget": [1000] * 6,
            "allocation_ratio": [0.05, 0.05, 0.1, 0.1, 0.2, 0.2],
            "train_size": [50, 50, 100, 100, 200, 200],
            "validation_size": [100] * 6,
            "loss": ["sample_mean", "mse", "sample_mean", "mse", "sample_mean", "mse"],
            "method": [
                "sample_mean",
                "lora_mse+ppi_plus",
                "sample_mean",
                "lora_mse+ppi_plus",
                "sample_mean",
                "lora_mse+ppi_plus",
            ],
            "replication_id": [0, 0, 0, 0, 0, 0],
            "estimated_estimator_variance": [0.10, 0.05, 0.10, 0.04, 0.10, 0.06],
            "ci_length": [1.0, 0.7, 1.0, 0.6, 1.0, 0.8],
            "residual_variance": [None, 2.0, None, 1.5, None, 1.4],
            "sample_savings": [0.0, 0.5, 0.0, 0.6, 0.0, 0.4],
            "bias": [0.0] * 6,
            "rmse": [0.1] * 6,
        }
    )
    curve = summarize_allocation_curve(metrics)
    ppi_curve = curve[curve["method"] == "lora_mse+ppi_plus"]
    assert ppi_curve["normalized_estimated_variance"].tolist() == pytest.approx([0.5, 0.4, 0.6])

    oracle = summarize_oracle_allocations(curve)
    assert oracle.loc[0, "oracle_train_size"] == 100
    assert oracle.loc[0, "oracle_normalized_variance"] == pytest.approx(0.4)

    scaling, loso = summarize_scaling_laws(curve)
    assert scaling.loc[0, "loss"] == "mse"
    assert set(loso["heldout_train_size"]) == {50, 100, 200}
