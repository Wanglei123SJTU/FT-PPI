from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.experiments.upworthy_question_scaling_law import (
    FEATURE_COLS,
    OutcomeScale,
    build_population_split,
    build_scaling_split,
    compute_inference_setup,
    ifvarq_loss_from_residuals,
    load_upworthy_pairs,
    outcome_scale_from_h_scale,
    task_index_to_cell,
    train_ids_for_s,
)


def _toy_frame(n_h: int = 12, n_t: int = 10) -> pd.DataFrame:
    rows = []
    sample_id = 0
    for split, n in [("h_scale", n_h), ("target", n_t)]:
        for i in range(n):
            q = [-1.0, 0.0, 1.0][i % 3]
            numeric = [-1.0, 0.0, 1.0][(i + 1) % 3]
            simplicity = (i - n / 2) / 3.0
            common = ((i % 5) - 2) / 2.0
            length = ((i % 4) - 1.5) / 2.0
            y = 0.2 - 0.1 * q + 0.05 * numeric + 0.01 * simplicity
            rows.append(
                {
                    "sample_id": sample_id,
                    "pair_id": sample_id,
                    "split": split,
                    "headline_a": f"Can this headline {sample_id} work?",
                    "headline_b": f"This headline {sample_id} works",
                    "y_logit_ctr_diff": y,
                    "delta_QUESTION": q,
                    "delta_NUMERIC": numeric,
                    "delta_SIMPLICITY": simplicity,
                    "delta_COMMON": common,
                    "delta_LENGTH": length,
                }
            )
            sample_id += 1
    return pd.DataFrame(rows)


def _config() -> dict:
    return {
        "seed": 123,
        "train_pool_size": 8,
        "validation_stop_size": 2,
        "validation_scale_size": 2,
        "s_grid": [2, 4, 8],
        "replication_ids": [0, 1],
        "methods": [
            {"name": "mse_stop_mse", "loss": "mse", "early_stopping_metric": "mse"},
            {"name": "ifvarq_stop_ifvarq", "loss": "ifvarq", "early_stopping_metric": "ifvarq"},
        ],
    }


def test_load_upworthy_pairs_and_population_split():
    path = Path(".pytest_tmp/upworthy_question_scaling_toy.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    _toy_frame().drop(columns=["sample_id"]).to_csv(path, index=False)
    df = load_upworthy_pairs(path)
    assert "sample_id" in df
    assert set(FEATURE_COLS).issubset(df.columns)
    population = build_population_split(df)
    assert len(population.h_scale_ids) == 12
    assert len(population.p_target_ids) == 10


def test_scaling_split_is_disjoint_and_nested():
    df = _toy_frame()
    split = build_scaling_split(df, _config(), replication_id=0)
    assert len(set(split.train_pool_ids) & set(split.v_stop_ids)) == 0
    assert len(set(split.train_pool_ids) & set(split.v_scale_ids)) == 0
    assert set(train_ids_for_s(split, 2)).issubset(set(train_ids_for_s(split, 4)))
    assert set(train_ids_for_s(split, 4)).issubset(set(train_ids_for_s(split, 8)))


def test_inference_setup_uses_target_hessian_and_question_weight():
    df = _toy_frame(n_h=12, n_t=12)
    population = build_population_split(df)
    y_scale = outcome_scale_from_h_scale(df, population.h_scale_ids)
    out, info = compute_inference_setup(df, population.p_target_ids, y_scale)
    assert "if_weight_question" in out
    assert info["target_feature"] == "delta_QUESTION"
    assert info["target_feature_index"] == 1
    assert np.isclose(info["question_beta_raw"], -0.1)
    assert info["direct_ols_ifvar_target_raw"] < 1e-20


def test_ifvarq_loss_is_weighted_residual_variance():
    residual = np.array([1.0, 2.0, 4.0, 8.0])
    weight = np.array([0.5, 1.0, -1.0, 2.0])
    weighted = residual * weight
    expected = np.mean((weighted - weighted.mean()) ** 2)
    assert np.isclose(ifvarq_loss_from_residuals(residual, weight), expected)


def test_task_index_to_cell_orders_methods_then_reps_then_s():
    config = _config()
    assert task_index_to_cell(config, 0) == ("mse_stop_mse", 0, 2)
    assert task_index_to_cell(config, 2) == ("mse_stop_mse", 0, 8)
    assert task_index_to_cell(config, 3) == ("mse_stop_mse", 1, 2)
    assert task_index_to_cell(config, 6) == ("ifvarq_stop_ifvarq", 0, 2)
