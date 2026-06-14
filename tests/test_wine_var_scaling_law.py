from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.wine_var_scaling_law import (
    SplitBundle,
    build_replication_splits,
    discrete_objective,
    fit_scaling_law,
    max_steps_for_s,
    raw_y,
    replay_rampup,
    scaled_y,
    task_index_to_rep_s,
    train_ids_for_s,
    validate_split_bundle,
    var_loss_from_residuals,
)


def _fake_clean(n: int = 25000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": np.arange(n, dtype=np.int64),
            "description": [f"review {i}" for i in range(n)],
            "points": 80.0 + (np.arange(n) % 21),
        }
    )


def _config() -> dict:
    return {
        "seed": 123,
        "budget_B": 10000,
        "validation_stop_size": 1000,
        "validation_scale_size": 1000,
        "eval_size": 2000,
        "replication_ids": [0, 1, 2],
        "s_grid": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    }


def test_replication_splits_are_disjoint_and_nested():
    clean = _fake_clean()
    config = _config()
    bundle = build_replication_splits(clean, config, replication_id=3)
    validate_split_bundle(bundle, config["s_grid"])

    assert len(bundle.l_ids) == 10000
    assert len(bundle.v_stop_ids) == 1000
    assert len(bundle.v_scale_ids) == 1000
    assert len(bundle.l_prime_ids) == 8000
    assert len(bundle.e_eval_ids) == 2000
    assert set(bundle.e_eval_ids).isdisjoint(set(bundle.l_ids))
    assert set(bundle.v_stop_ids).issubset(set(bundle.l_ids))
    assert set(bundle.v_scale_ids).issubset(set(bundle.l_ids))
    assert set(bundle.v_stop_ids).isdisjoint(set(bundle.v_scale_ids))

    previous: set[int] = set()
    for s in config["s_grid"]:
        current = set(train_ids_for_s(bundle, s))
        assert previous.issubset(current)
        assert current.issubset(set(bundle.l_prime_ids))
        previous = current


def test_validate_split_bundle_rejects_eval_leakage():
    bundle = SplitBundle(
        l_ids=np.array([1, 2, 3, 4]),
        v_stop_ids=np.array([1]),
        v_scale_ids=np.array([2]),
        l_prime_ids=np.array([3, 4]),
        train_order_ids=np.array([3, 4]),
        e_eval_ids=np.array([4, 5]),
    )
    try:
        validate_split_bundle(bundle, [1, 2])
    except ValueError as exc:
        assert "E_eval overlaps L" in str(exc)
    else:
        raise AssertionError("expected leakage validation failure")


def test_fixed_label_scaling_round_trips():
    y = np.array([80.0, 90.0, 100.0])
    scaled = scaled_y(y)
    assert np.allclose(scaled, [-2.0, 0.0, 2.0])
    assert np.allclose(raw_y(scaled), y)


def test_var_loss_matches_batch_residual_variance_mean_form():
    residual = np.array([1.0, 2.0, 4.0, 5.0])
    expected = np.mean((residual - residual.mean()) ** 2)
    assert np.isclose(var_loss_from_residuals(residual), expected)


def test_max_steps_uses_actual_batch_size():
    assert max_steps_for_s(100, 256, max_epochs=20) == 20
    assert max_steps_for_s(1000, 256, max_epochs=20) == 80
    assert max_steps_for_s(1000, 128, max_epochs=20) == 160


def test_task_index_mapping():
    config = _config()
    assert task_index_to_rep_s(config, 0) == (0, 100)
    assert task_index_to_rep_s(config, 9) == (0, 1000)
    assert task_index_to_rep_s(config, 10) == (1, 100)
    assert task_index_to_rep_s(config, 29) == (2, 1000)


def test_scaling_law_fit_respects_bounds_on_synthetic_curve():
    s = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=float)
    y = 3.0 * s ** (-0.4) + 0.25
    fit = fit_scaling_law(s, y, population_var_y=2.0)
    assert fit["a"] > 0
    assert 0.02 <= fit["alpha"] <= 1.5
    assert 0.0 <= fit["b"] <= 2.0
    assert fit["r2"] > 0.99


def test_rampup_replay_stops_and_uses_best_seen_not_current_largest():
    rows = []
    s_grid = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    validation_var = [8.0, 5.0, 3.0, 2.2, 2.0, 2.1, 2.3, 2.6, 3.0, 3.6]
    for s, v_var in zip(s_grid, validation_var):
        rows.append(
            {
                "replication_id": 0,
                "s_train": s,
                "n_eff": 8000,
                "population_var_y_scaled": 10.0,
                "validation_scale_residual_var": v_var,
            }
        )
    metrics = pd.DataFrame(rows)
    ramp = replay_rampup(metrics, min_points_for_stop=4)
    assert len(ramp) == 1
    assert ramp.loc[0, "stopped_stage_index"] >= 4
    assert ramp.loc[0, "oracle_source"] == "validation_scale"
    assert ramp.loc[0, "ramp_s_best_seen"] in set(s_grid)
    assert ramp.loc[0, "oracle_s"] == min(
        s_grid,
        key=lambda s: discrete_objective(validation_var[s_grid.index(s)], s, 8000),
    )
    assert ramp.loc[0, "regret"] >= 0.0
