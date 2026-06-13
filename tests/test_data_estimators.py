from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.build_wine import clean_wine, make_split_roles, sample_population
from src.estimators.mean_ppi import estimate_ppi_plus_lambda, ppi_estimate, ppi_plus_estimate
from src.eval.predict import save_prediction_frame
from src.train.loss_utils import residual_variance_loss_numpy


def test_clean_wine_deduplicates_description(tmp_path):
    csv = tmp_path / "wine.csv"
    pd.DataFrame(
        {
            "description": ["a", "b", "a", None],
            "points": [90, 91, 92, 93],
            "unused": [1, 2, 3, 4],
        }
    ).to_csv(csv, index=False)
    clean = clean_wine(csv)
    assert len(clean) == 2
    assert clean["description"].is_unique
    assert clean["points"].tolist() == [90.0, 91.0]


def test_split_roles_are_disjoint_and_sized():
    df = pd.DataFrame({"sample_id": np.arange(20), "description": [str(i) for i in range(20)], "points": 90.0})
    roles = make_split_roles(df, budget=8, train_size=3, validation_size=2, seed=123)
    assert roles["sample_id"].is_unique
    assert roles["split_role"].value_counts().to_dict() == {
        "unlabeled": 12,
        "correction": 3,
        "train": 3,
        "validation": 2,
    }


def test_sample_population_is_reproducible():
    df = pd.DataFrame({"sample_id": np.arange(100), "description": [str(i) for i in range(100)], "points": 90.0})
    a = sample_population(df, 10, seed=7)
    b = sample_population(df, 10, seed=7)
    assert a["sample_id"].tolist() == b["sample_id"].tolist()


def test_ppi_with_zero_predictor_is_correction_mean():
    y = np.array([88.0, 90.0, 92.0])
    f_c = np.zeros_like(y)
    f_u = np.zeros(10)
    result = ppi_estimate(y, f_c, f_u)
    assert result.estimate == np.mean(y)


def test_ppi_plus_lambda_falls_back_when_predictor_variance_zero():
    y = np.array([88.0, 90.0, 92.0])
    f_c = np.ones_like(y)
    lam = estimate_ppi_plus_lambda(y, f_c, n_unlabeled=10)
    result = ppi_plus_estimate(y, f_c, np.ones(10))
    assert np.isfinite(lam)
    assert lam == 1.0
    assert np.isfinite(result.estimate)


def test_residual_variance_loss_numpy_matches_formula():
    y = np.array([1.0, 2.0, 4.0])
    pred = np.array([0.5, 1.5, 3.5])
    residual = y - pred
    expected = np.mean((residual - residual.mean()) ** 2)
    assert residual_variance_loss_numpy(y, pred) == expected


def test_prediction_parquet_contains_required_columns(tmp_path):
    df = pd.DataFrame(
        {
            "sample_id": [1, 2],
            "split_role": ["correction", "unlabeled"],
            "points": [88.0, 90.0],
        }
    )
    path = tmp_path / "predictions.parquet"
    save_prediction_frame(path, df, [0.0, 1.0], label_mean=88.0, label_std=2.0, method="m", model_name="model", loss="mse")
    out = pd.read_parquet(path)
    assert {"sample_id", "split_role", "y_true", "pred_scaled", "pred_mean"}.issubset(out.columns)
    assert out["pred_mean"].tolist() == [88.0, 90.0]
