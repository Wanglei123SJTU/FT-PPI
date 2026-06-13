from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.allocation_regret import summarize_rampup_allocation_regret


def test_rampup_allocation_regret_selects_low_regret_grid_point():
    curve = pd.DataFrame(
        {
            "budget": [1000] * 4,
            "allocation_ratio": [0.05, 0.10, 0.20, 0.30],
            "train_size": [50, 100, 200, 300],
            "validation_size": [100] * 4,
            "loss": ["mse"] * 4,
            "method": ["lora_mse+ppi_plus"] * 4,
            "mean_residual_variance": [12.0, 8.0, 6.0, 5.5],
            "mean_estimated_variance": [0.020, 0.012, 0.010, 0.011],
            "normalized_estimated_variance": [2.0, 1.2, 1.0, 1.1],
            "sample_mean_variance": [0.010] * 4,
        }
    )

    regret = summarize_rampup_allocation_regret(curve, rampup_points=3)

    assert len(regret) == 1
    assert regret.loc[0, "oracle_train_size"] == 200
    assert regret.loc[0, "selected_train_size"] in {200, 300}
    assert regret.loc[0, "relative_regret"] >= 0
    assert not bool(regret.loc[0, "beats_sample_mean"])


def test_rampup_allocation_regret_requires_three_rampup_points():
    curve = pd.DataFrame(
        {
            "budget": [1000] * 2,
            "allocation_ratio": [0.05, 0.10],
            "train_size": [50, 100],
            "validation_size": [100] * 2,
            "loss": ["mse"] * 2,
            "method": ["lora_mse+ppi_plus"] * 2,
            "mean_residual_variance": [12.0, 8.0],
            "mean_estimated_variance": [0.020, 0.012],
            "normalized_estimated_variance": [2.0, 1.2],
            "sample_mean_variance": [0.010] * 2,
        }
    )

    regret = summarize_rampup_allocation_regret(curve, rampup_points=3)

    assert pd.isna(regret.loc[0, "fit_alpha"])
    assert regret.loc[0, "selected_train_size"] == 50


def test_rampup_allocation_regret_validates_rampup_count():
    with pytest.raises(ValueError, match="at least 3"):
        summarize_rampup_allocation_regret(pd.DataFrame(), rampup_points=2)
