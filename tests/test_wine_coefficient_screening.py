from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.experiments.wine_coefficient_screening import (
    Y_COL,
    build_budget_win_table,
    build_wine_screening_frame,
    compute_candidate_summary,
    compute_tfidf_scaling,
    fit_ols_stats,
    positive_ifvar_weights,
)


TEST_TMP = Path("artifacts/test_tmp/wine_coefficient_screening")


def _toy_wine_csv(path, n: int = 80) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        expensive = i % 2 == 0
        text_quality = (i // 2) % 2 == 0
        good_word = ("velvety elegant " if expensive else "thin simple ") + ("brilliant" if text_quality else "dull")
        country = "US" if i % 3 == 0 else "France"
        variety = "Pinot Noir" if i % 4 == 0 else "Chardonnay"
        price = 50.0 if expensive else 10.0
        points = 90.0 + (1.5 if expensive else -1.5) + (4.0 if text_quality else -4.0)
        rows.append(
            {
                "description": f"{good_word} wine sample {i}",
                "points": points,
                "price": price,
                "country": country,
                "variety": variety,
                "title": f"wine {i}",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _toy_path(name: str) -> Path:
    return TEST_TMP / name


def test_build_wine_screening_frame_has_splits_and_features():
    path = _toy_path("wine_splits.csv")
    _toy_wine_csv(path, n=80)
    frame, infos, summary = build_wine_screening_frame(
        path,
        seed=1,
        experimental_population_size=60,
        h_scale_size=25,
        target_size=20,
        countries=["US", "France"],
        varieties=["Pinot Noir", "Chardonnay"],
    )
    assert len(frame) == 60
    assert (frame["split"] == "h_scale").sum() == 25
    assert (frame["split"] == "target").sum() == 20
    assert "feat_log_price" in frame.columns
    assert "feat_country_us" in infos
    assert summary["y_scaling"] == "points_scaled = (points - 90) / 5"


def test_fit_ols_stats_recovers_scaled_price_signal():
    path = _toy_path("wine_ols.csv")
    _toy_wine_csv(path, n=80)
    frame, _, _ = build_wine_screening_frame(
        path,
        seed=2,
        experimental_population_size=60,
        h_scale_size=20,
        target_size=30,
        countries=["US"],
        varieties=["Pinot Noir"],
    )
    target = frame.loc[frame["split"] == "target"]
    stats = fit_ols_stats(target, "feat_log_price", ["feat_log_price"])
    assert stats.beta_raw_points > 0.2
    assert np.isfinite(stats.ifvar)
    assert stats.if_weight_p99 < 10.0


def test_positive_ifvar_weights_are_positive_and_normalized():
    weights = positive_ifvar_weights(np.array([-2.0, -1.0, 0.0, 1.0, 5.0]), clip_quantile=0.8)
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[-1] < 10.0


def test_budget_win_table_uses_direct_label_baseline():
    scaling = pd.DataFrame(
        {
            "feature": ["feat_signal", "feat_signal", "feat_signal"],
            "training_objective": ["mse", "mse", "mse"],
            "s": [0, 100, 250],
            "ifvar": [1.0, 0.5, 0.4],
        }
    )
    summary = pd.DataFrame({"feature": ["feat_signal"], "direct_ifvar": [1.0]})
    out = build_budget_win_table(scaling, summary, budgets=[500])
    row = out.iloc[0]
    assert bool(row["wins"])
    assert row["best_s"] == 100
    assert np.isclose(row["variance_ratio_vs_direct"], 0.625)


def test_tfidf_scaling_reduces_ifvar_on_toy_signal():
    path = _toy_path("wine_tfidf.csv")
    _toy_wine_csv(path, n=120)
    frame, infos, _ = build_wine_screening_frame(
        path,
        seed=3,
        experimental_population_size=100,
        h_scale_size=60,
        target_size=30,
        countries=["US"],
        varieties=["Pinot Noir"],
    )
    features = ["feat_log_price"]
    summary, stats = compute_candidate_summary(frame, features, infos, budgets=[500])
    assert summary.loc[0, "abs_beta_raw_points"] > 2.0
    scaling = compute_tfidf_scaling(
        frame,
        features,
        stats,
        s_grid=[0, 10, 20],
        replications=2,
        seed=4,
        train_pool_size=25,
        validation_stop_size=15,
        validation_scale_size=15,
        alphas=[0.1, 1.0],
        max_tfidf_features=100,
        min_df=1,
        if_weight_clip_quantile=0.99,
    )
    constant = scaling.loc[scaling["training_objective"] == "constant", "ifvar"].mean()
    curve = scaling.loc[scaling["training_objective"] == "mse"].groupby("s")["ifvar"].mean()
    assert curve.loc[20] < constant
    assert frame[Y_COL].notna().all()
