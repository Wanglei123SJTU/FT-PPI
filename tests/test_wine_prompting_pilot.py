from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.wine_prompting_pilot import (
    parse_rating,
    ppi_variance,
    ppiplus_lambda,
    ppiplus_variance,
    projection_table,
    summarize_bootstrap,
    variance_components,
)


def test_parse_rating_accepts_json_and_text():
    assert parse_rating('{"rating": 91}') == 91.0
    assert parse_rating("Rating: 87") == 87.0
    assert parse_rating("The predicted score is 94.") == 94.0
    assert parse_rating('{"rating": 103}') is None
    assert parse_rating("no rating returned") is None


def test_variance_components_match_manual_values():
    pred = pd.DataFrame(
        {
            "y_raw": [88.0, 90.0, 92.0, 94.0],
            "pred_raw": [87.0, 91.0, 91.0, 95.0],
        }
    )
    pred["residual_raw"] = pred["y_raw"] - pred["pred_raw"]
    comp = variance_components(pred)
    y = pred["y_raw"].to_numpy()
    f = pred["pred_raw"].to_numpy()
    residual = y - f
    assert np.isclose(comp.y_var, np.var(y, ddof=1))
    assert np.isclose(comp.pred_var, np.var(f, ddof=1))
    assert np.isclose(comp.residual_var, np.var(residual, ddof=1))
    assert np.isclose(comp.covariance_y_pred, np.cov(y, f, ddof=1)[0, 1])


def test_ppi_variance_formula():
    assert np.isclose(ppi_variance(4.0, 9.0, correction_n=100, unlabeled_n=900), 4.0 / 100 + 9.0 / 900)


def test_ppiplus_lambda_and_variance_are_finite():
    y = np.array([88.0, 90.0, 92.0, 94.0, 96.0])
    f = np.array([87.0, 90.0, 91.0, 95.0, 95.0])
    pred_var = np.var(f, ddof=1)
    cov = np.cov(y, f, ddof=1)[0, 1]
    lam = ppiplus_lambda(cov, pred_var, correction_n=100, unlabeled_n=1000)
    var, lam2, adjusted = ppiplus_variance(y, f, correction_n=100, unlabeled_n=1000)
    assert np.isfinite(lam)
    assert np.isclose(lam, lam2)
    assert np.isfinite(var)
    assert np.isfinite(adjusted)


def test_projection_summary_contains_ppi_and_ppiplus():
    target = pd.DataFrame({"points": [88.0, 90.0, 92.0, 94.0, 96.0, 98.0]})
    predictions = pd.DataFrame(
        {
            "y_raw": [88.0, 90.0, 92.0, 94.0, 96.0],
            "pred_raw": [87.0, 90.0, 91.0, 95.0, 95.0],
        }
    )
    point, boot = projection_table(
        target=target,
        predictions=predictions,
        budgets=[3],
        validation_fraction=0.2,
        bootstrap_reps=5,
        seed=123,
    )
    summary = summarize_bootstrap(point, boot)
    assert set(summary["method"]) == {"PPI-Only", "PPI++-Only"}
    assert (summary["projected_var"] > 0).all()
    assert summary["projected_var_se"].notna().all()
