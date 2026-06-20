from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.upworthy_feature_screening import (
    DEFAULT_CORE_CONTROLS,
    build_budget_win_table,
    compute_candidate_summaries,
    compute_tfidf_scaling,
    core_controls_for_target,
    discover_candidate_features,
    fit_text_featurizer,
    fit_surrogate_featurizer,
    fit_tfidf_vectorizer,
    pair_tfidf_matrix,
    surrogate_matrix,
)


def _toy_screening_frame(n_h: int = 40, n_t: int = 24) -> pd.DataFrame:
    rows = []
    sample_id = 0
    for split, n in [("h_scale", n_h), ("target", n_t)]:
        for i in range(n):
            signal = 1.0 if i % 2 == 0 else -1.0
            bonus = 1.0 if (i // 2) % 2 == 0 else -1.0
            length = ((i % 5) - 2.0) / 2.0
            common = ((i % 7) - 3.0) / 3.0
            a_words = ["headline"]
            b_words = ["headline"]
            (a_words if signal > 0 else b_words).append("magic")
            (a_words if bonus > 0 else b_words).append("bonus")
            headline_a = " ".join(a_words)
            headline_b = " ".join(b_words)
            y = 1.5 * signal + 0.5 * bonus
            rows.append(
                {
                    "sample_id": sample_id,
                    "pair_id": sample_id,
                    "split": split,
                    "headline_a": headline_a,
                    "headline_b": headline_b,
                    "y_logit_ctr_diff": y,
                    "clicks_a": 10,
                    "clicks_b": 9,
                    "impressions_a": 1000,
                    "impressions_b": 1000,
                    "delta_SIGNAL": signal,
                    "delta_SIGNAL_raw": signal,
                    "delta_SIGNAL_scale": 1.0,
                    "delta_LENGTH": length,
                    "delta_SIMPLICITY": -length,
                    "delta_COMMON": common,
                    "delta_QUESTION": 0.0,
                    "delta_NUMERIC": 0.0,
                    "delta_VADER_COMPOUND": signal / 2.0,
                }
            )
            sample_id += 1
    return pd.DataFrame(rows)


def test_discover_candidate_features_excludes_raw_and_scale():
    df = _toy_screening_frame()
    candidates = discover_candidate_features(df)
    assert "delta_SIGNAL" in candidates
    assert "delta_SIGNAL_raw" not in candidates
    assert "delta_SIGNAL_scale" not in candidates


def test_pair_tfidf_matrix_is_antisymmetric_under_swap():
    frame = pd.DataFrame(
        {
            "headline_a": ["magic useful headline"],
            "headline_b": ["plain headline"],
        }
    )
    swapped = frame.rename(columns={"headline_a": "headline_b", "headline_b": "headline_a"})
    vectorizer = fit_tfidf_vectorizer(pd.concat([frame, swapped], ignore_index=True), max_features=100, min_df=1)
    x = pair_tfidf_matrix(vectorizer, frame)
    x_swapped = pair_tfidf_matrix(vectorizer, swapped)
    assert np.allclose((x + x_swapped).toarray(), 0.0)


def test_combined_text_featurizer_is_antisymmetric_under_swap():
    frame = pd.DataFrame(
        {
            "headline_a": ["magic useful headline"],
            "headline_b": ["plain headline"],
        }
    )
    swapped = frame.rename(columns={"headline_a": "headline_b", "headline_b": "headline_a"})
    vectorizer = fit_text_featurizer(
        pd.concat([frame, swapped], ignore_index=True),
        mode="word_char",
        max_features=100,
        min_df=1,
    )
    x = pair_tfidf_matrix(vectorizer, frame)
    x_swapped = pair_tfidf_matrix(vectorizer, swapped)
    assert np.allclose((x + x_swapped).toarray(), 0.0)


def test_structured_surrogate_uses_delta_columns():
    frame = pd.DataFrame(
        {
            "headline_a": ["a", "b"],
            "headline_b": ["b", "a"],
            "delta_SIGNAL": [2.0, -2.0],
            "delta_LENGTH": [0.5, -0.5],
        }
    )
    featurizer = fit_surrogate_featurizer(
        frame,
        mode="structured",
        max_features=100,
        min_df=1,
        structured_columns=["delta_SIGNAL", "delta_LENGTH"],
    )
    x = surrogate_matrix(featurizer, frame).toarray()
    assert np.allclose(x[0], -x[1])
    assert np.allclose(x[0], [2.0, 0.5])


def test_length_readability_target_removes_length_controls():
    controls = core_controls_for_target("delta_READING_EASE", DEFAULT_CORE_CONTROLS)
    assert "delta_SIMPLICITY" not in controls
    assert "delta_LENGTH" not in controls
    assert "delta_COMMON" in controls


def test_budget_win_inequality():
    scaling = pd.DataFrame(
        {
            "delta_col": ["delta_SIGNAL", "delta_SIGNAL", "delta_SIGNAL"],
            "feature": ["SIGNAL", "SIGNAL", "SIGNAL"],
            "s": [0, 100, 250],
            "ifvar": [1.0, 0.7, 0.6],
            "ifvar_ratio_to_s0": [1.0, 0.7, 0.6],
        }
    )
    out = build_budget_win_table(scaling, budgets=[500])
    row = out.iloc[0]
    assert bool(row["wins"])
    assert row["best_s"] == 100
    assert np.isclose(row["win_threshold_at_best_s"], 0.8)


def test_toy_text_signal_reduces_target_if_variance():
    df = _toy_screening_frame()
    candidates = ["delta_SIGNAL"]
    summary, _, controlled = compute_candidate_summaries(df, candidates, [], budgets=[500])
    assert summary.loc[0, "controlled_abs_beta"] > 1.0
    scaling = compute_tfidf_scaling(
        df,
        candidates,
        controlled,
        s_values=[0, 8, 16],
        replications=2,
        seed=123,
        train_pool_size=20,
        validation_stop_size=8,
        validation_scale_size=8,
        alphas=[0.1, 1.0],
        max_features=100,
        min_df=1,
    )
    ratios = scaling.groupby("s")["ifvar_ratio_to_s0"].mean()
    assert ratios.loc[16] < ratios.loc[0]
