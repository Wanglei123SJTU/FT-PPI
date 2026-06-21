from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.data.prepare_helpsteer2_preference import (
    build_helpsteer2_preference_frame,
    has_structured_format,
    prompt_coverage,
)
from src.experiments.helpsteer2_preference_regression import (
    build_budget_ci_proxy,
    build_regression_summary,
    fit_ols_result,
)
from src.experiments.helpsteer2_embedding_extraction import pairwise_difference
from src.experiments.helpsteer2_embedding_mlp_scaling import antisymmetric_from_scores


def _toy_preference_frame(n: int = 80) -> pd.DataFrame:
    rows = []
    for i in range(n):
        prompt = "Explain customer retention strategy with examples"
        structured = i % 2 == 0
        if structured:
            response_1 = "Segment customers by lifecycle, identify churn signals, and offer targeted incentives."
            response_2 = "- Segment customers by lifecycle\n- Identify churn signals\n- Offer targeted incentives"
        else:
            response_1 = "Segment customers by lifecycle, identify churn signals, and offer targeted incentives."
            response_2 = "Segment customers by lifecycle, identify churn signals, and offer targeted incentives."
        rows.append(
            {
                "split": "train",
                "prompt": prompt,
                "response_1": response_1,
                "response_2": response_2,
                "preference_strength": 1.5 * structured + 0.8,
            }
        )
    return pd.DataFrame(rows)


def test_prompt_coverage_uses_prompt_tokens():
    coverage = prompt_coverage(
        "Design a customer retention strategy",
        "A retention strategy keeps customers active.",
    )
    assert coverage > 0.0
    assert coverage <= 1.0
    assert prompt_coverage("", "anything") == 0.0


def test_has_structured_format_detects_bullets_numbers_and_markdown():
    assert has_structured_format("1. First step\n2. Second step") == 1
    assert has_structured_format("- First\n- Second") == 1
    assert has_structured_format("**Important:** answer") == 1
    assert has_structured_format("Plain sentence only.") == 0


def test_build_helpsteer2_preference_frame_sign_convention():
    raw = pd.DataFrame(
        [
            {
                "split": "train",
                "prompt": "List retention tactics",
                "response_1": "Keep users.",
                "response_2": "- Email reminders\n- Loyalty offers\n- Better onboarding",
                "preference_strength": 2,
            }
        ]
    )
    frame, summary = build_helpsteer2_preference_frame(raw)
    row = frame.iloc[0]
    assert row["y_preference_strength"] == 2
    assert row["delta_log_length"] > 0
    assert row["delta_format"] == 1
    assert "response_2 minus response_1" in summary["sign_convention"]


def test_fit_ols_result_recovers_toy_format_signal():
    raw = _toy_preference_frame(n=100)
    frame, _ = build_helpsteer2_preference_frame(raw)
    result = fit_ols_result(
        frame,
        target="delta_format",
        feature_columns=["delta_log_length", "delta_prompt_coverage", "delta_format"],
        model="controlled",
    )
    assert result.beta > 0.5
    assert result.nonzero_share > 0.4
    assert np.isfinite(result.ifvar)
    assert result.if_weight_p99 < 50.0


def test_regression_summary_and_ci_proxy_include_targets():
    raw = _toy_preference_frame(n=100)
    frame, _ = build_helpsteer2_preference_frame(raw)
    summary = build_regression_summary(frame)
    controlled = summary.loc[summary["model"] == "controlled"]
    assert set(controlled["target"]) == {"delta_log_length", "delta_prompt_coverage", "delta_format"}
    ci = build_budget_ci_proxy(summary, budgets=[500])
    assert set(ci["budget"]) == {500}
    row = ci.iloc[0]
    source = summary.loc[(summary["model"] == row["model"]) & (summary["target"] == row["target"])].iloc[0]
    expected = 2.0 * 1.96 * math.sqrt(source["ifvar"] / 500)
    assert math.isclose(row["ci95_length_proxy"], expected)


def test_pairwise_difference_is_response2_minus_response1():
    emb_1 = np.array([[1.0, 2.0], [3.0, 1.0]])
    emb_2 = np.array([[4.0, 1.0], [2.0, 5.0]])
    diff = pairwise_difference(emb_1, emb_2)
    swapped = pairwise_difference(emb_2, emb_1)
    assert np.allclose(diff, np.array([[3.0, -1.0], [-1.0, 4.0]]))
    assert np.allclose(swapped, -diff)


def test_antisymmetric_scores_flip_sign_under_swap():
    score_h = np.array([3.0, -2.0, 0.5])
    score_neg_h = np.array([1.0, 4.0, -0.5])
    yhat = antisymmetric_from_scores(score_h, score_neg_h)
    yhat_swapped = antisymmetric_from_scores(score_neg_h, score_h)
    assert np.allclose(yhat, np.array([1.0, -3.0, 0.5]))
    assert np.allclose(yhat_swapped, -yhat)
