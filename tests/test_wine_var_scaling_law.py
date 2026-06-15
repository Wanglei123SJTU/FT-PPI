from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.wine_full_grid_allocation import (
    allocation_correction_ids_for_rho,
    allocation_train_ids_for_rho,
    build_allocation_split,
    ppi_metrics,
    task_index_to_allocation_cell,
    theoretical_allocation,
)
from src.experiments.wine_var_scaling_law import (
    SplitBundle,
    build_population_split,
    build_replication_splits,
    configured_losses,
    configured_methods,
    discrete_objective,
    fit_scaling_law,
    max_steps_for_s,
    mse_loss_from_residuals,
    raw_y,
    replay_rampup,
    scaled_y,
    stopping_value_from_metrics,
    task_index_to_loss_rep_s,
    task_index_to_rep_s,
    train_ids_for_s,
    training_loss_from_residuals,
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
        "experimental_population_size": 25000,
        "h_scale_size": 10000,
        "split_source": "h_scale",
        "budget_B": 10000,
        "validation_stop_size": 1000,
        "validation_scale_size": 1000,
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
    assert set(bundle.v_stop_ids).issubset(set(bundle.l_ids))
    assert set(bundle.v_scale_ids).issubset(set(bundle.l_ids))
    assert set(bundle.v_stop_ids).isdisjoint(set(bundle.v_scale_ids))
    assert set(bundle.l_prime_ids).isdisjoint(set(bundle.v_stop_ids))
    assert set(bundle.l_prime_ids).isdisjoint(set(bundle.v_scale_ids))

    previous: set[int] = set()
    for s in config["s_grid"]:
        current = set(train_ids_for_s(bundle, s))
        assert previous.issubset(current)
        assert current.issubset(set(bundle.l_prime_ids))
        previous = current


def test_population_split_matches_paper_protocol_for_scaling_validation():
    clean = _fake_clean()
    config = _config()
    population = build_population_split(clean, config)
    assert len(population.p0_ids) == 25000
    assert len(population.h_scale_ids) == 10000
    assert len(population.p_target_ids) == 15000
    assert set(population.h_scale_ids).isdisjoint(set(population.p_target_ids))
    assert set(population.h_scale_ids) | set(population.p_target_ids) == set(population.p0_ids)

    rep0 = build_replication_splits(clean, config, replication_id=0)
    rep1 = build_replication_splits(clean, config, replication_id=1)
    assert set(rep0.l_ids) == set(population.h_scale_ids)
    assert set(rep1.l_ids) == set(population.h_scale_ids)
    assert set(rep0.l_ids) == set(rep1.l_ids)
    assert set(rep0.v_stop_ids).isdisjoint(set(rep0.v_scale_ids))
    assert set(rep1.v_stop_ids).isdisjoint(set(rep1.v_scale_ids))
    assert set(rep0.v_stop_ids) != set(rep1.v_stop_ids)


def test_target_split_draws_labeled_budget_from_p_target():
    clean = _fake_clean()
    config = dict(_config())
    config["split_source"] = "p_target"
    config["budget_B"] = 500
    config["validation_stop_size"] = 100
    config["validation_scale_size"] = 100
    config["s_grid"] = [50, 100, 200, 300]

    population = build_population_split(clean, config)
    bundle = build_replication_splits(clean, config, replication_id=0)
    assert len(bundle.l_ids) == 500
    assert set(bundle.l_ids).issubset(set(population.p_target_ids))
    assert set(bundle.l_ids).isdisjoint(set(population.h_scale_ids))


def test_full_grid_allocation_split_uses_target_population_and_nested_corrections():
    clean = _fake_clean()
    config = dict(_config())
    config.update(
        {
            "split_source": "p_target",
            "budgets": [300, 500],
            "validation_fraction": 0.2,
            "allocation_grid": [0.025, 0.05, 0.10, 0.50, 0.90],
            "scaling_law_params_raw": {"a": 17.4, "alpha": 0.57, "b": 2.83},
        }
    )
    population = build_population_split(clean, config)
    split = build_allocation_split(clean, config, budget_B=500, replication_id=0)
    assert len(split.labeled_ids) == 500
    assert len(split.validation_ids) == 100
    assert len(split.effective_ids) == 400
    assert len(split.unlabeled_ids) == 14500
    assert set(split.labeled_ids).issubset(set(population.p_target_ids))
    assert set(split.labeled_ids).isdisjoint(set(population.h_scale_ids))

    previous: set[int] = set()
    for rho in config["allocation_grid"]:
        train = set(allocation_train_ids_for_rho(split, rho))
        correction = set(allocation_correction_ids_for_rho(split, rho))
        assert previous.issubset(train)
        assert train.isdisjoint(correction)
        assert train | correction == set(split.effective_ids)
        previous = train

    theory = theoretical_allocation(config, 500)
    assert 0.0 < theory["theory_rho"] < 1.0
    assert theory["theory_eval_rho"] in config["allocation_grid"]


def test_full_grid_task_index_mapping():
    config = dict(_config())
    config.update(
        {
            "split_source": "p_target",
            "budgets": [300, 500],
            "allocation_grid": [0.025, 0.10],
            "methods": [
                {"name": "mse_stop_mse", "loss": "mse", "early_stopping_metric": "mse"},
                {"name": "var_stop_var", "loss": "var", "early_stopping_metric": "var"},
            ],
        }
    )
    assert task_index_to_allocation_cell(config, 0) == ("mse_stop_mse", 300, 0, 0.025)
    assert task_index_to_allocation_cell(config, 1) == ("mse_stop_mse", 300, 0, 0.10)
    assert task_index_to_allocation_cell(config, 6) == ("mse_stop_mse", 500, 0, 0.025)
    assert task_index_to_allocation_cell(config, 12) == ("var_stop_var", 300, 0, 0.025)


def test_ppi_metrics_uses_correction_and_unlabeled_predictions():
    clean = _fake_clean(20)
    p_target_ids = np.arange(20)
    labeled_ids = np.array([0, 1, 2, 3])
    correction_pred = pd.DataFrame(
        {
            "sample_id": [2, 3],
            "y_raw": [82.0, 83.0],
            "y_scaled": [-1.6, -1.4],
            "pred_scaled": [-1.5, -1.5],
            "pred_raw": [82.5, 82.5],
            "residual_scaled": [-0.1, 0.1],
            "residual_raw": [-0.5, 0.5],
        }
    )
    unlabeled_pred = pd.DataFrame(
        {
            "sample_id": [4, 5, 6],
            "pred_scaled": [-1.0, -0.8, -0.6],
            "pred_raw": [85.0, 86.0, 87.0],
        }
    )
    metrics = ppi_metrics(clean, p_target_ids, labeled_ids, correction_pred, unlabeled_pred)
    assert np.isclose(metrics["mu_hat_ppi_raw"], 86.0)
    assert metrics["correction_size"] == 2
    assert metrics["unlabeled_size"] == 3
    assert metrics["ppi_var_est_raw"] > 0.0


def test_validate_split_bundle_rejects_validation_leakage():
    bundle = SplitBundle(
        l_ids=np.array([1, 2, 3, 4]),
        v_stop_ids=np.array([1]),
        v_scale_ids=np.array([2]),
        l_prime_ids=np.array([2, 3, 4]),
        train_order_ids=np.array([3, 4]),
    )
    try:
        validate_split_bundle(bundle, [1, 2])
    except ValueError as exc:
        assert "L_prime is not exactly L minus V_stop and V_scale" in str(exc)
    else:
        raise AssertionError("expected leakage validation failure")


def test_fixed_label_scaling_round_trips():
    y = np.array([80.0, 90.0, 100.0])
    scaled = scaled_y(y)
    assert np.allclose(scaled, [-2.0, 0.0, 2.0])
    assert np.allclose(raw_y(scaled), y)


def test_var_loss_matches_batch_residual_variance_mean_form():
    residual = np.array([1.0, 2.0, 4.0, 5.0])
    expected_var = np.mean((residual - residual.mean()) ** 2)
    expected_mse = np.mean(residual**2)
    assert np.isclose(var_loss_from_residuals(residual), expected_var)
    assert np.isclose(mse_loss_from_residuals(residual), expected_mse)
    assert np.isclose(training_loss_from_residuals(residual, "var"), expected_var)
    assert np.isclose(training_loss_from_residuals(residual, "mse"), expected_mse)


def test_stopping_value_can_use_variance_or_mse():
    metrics = {"residual_var_scaled": 2.0, "rmse_scaled": 3.0}
    assert stopping_value_from_metrics(metrics, "var") == 2.0
    assert stopping_value_from_metrics(metrics, "mse") == 9.0


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

    loss_config = dict(config)
    loss_config["losses"] = ["mse", "var"]
    assert configured_losses(loss_config) == ["mse", "var"]
    assert task_index_to_loss_rep_s(loss_config, 0) == ("mse", 0, 100)
    assert task_index_to_loss_rep_s(loss_config, 29) == ("mse", 2, 1000)
    assert task_index_to_loss_rep_s(loss_config, 30) == ("var", 0, 100)
    assert task_index_to_loss_rep_s(loss_config, 59) == ("var", 2, 1000)

    method_config = dict(config)
    method_config["methods"] = [
        {"name": "mse_stop_mse", "loss": "mse", "early_stopping_metric": "mse"},
        {"name": "mse_stop_var", "loss": "mse", "early_stopping_metric": "var"},
        {"name": "var_stop_var", "loss": "var", "early_stopping_metric": "var"},
    ]
    methods = configured_methods(method_config)
    assert [method.name for method in methods] == ["mse_stop_mse", "mse_stop_var", "var_stop_var"]
    assert [method.training_loss for method in methods] == ["mse", "mse", "var"]
    assert [method.early_stopping_metric for method in methods] == ["mse", "var", "var"]
    assert task_index_to_loss_rep_s(method_config, 0) == ("mse_stop_mse", 0, 100)
    assert task_index_to_loss_rep_s(method_config, 29) == ("mse_stop_mse", 2, 1000)
    assert task_index_to_loss_rep_s(method_config, 30) == ("mse_stop_var", 0, 100)
    assert task_index_to_loss_rep_s(method_config, 59) == ("mse_stop_var", 2, 1000)
    assert task_index_to_loss_rep_s(method_config, 60) == ("var_stop_var", 0, 100)
    assert task_index_to_loss_rep_s(method_config, 89) == ("var_stop_var", 2, 1000)


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
