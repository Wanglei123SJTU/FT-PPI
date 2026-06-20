from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.prepare_upworthy_pairs import clean_arm_level, make_one_pair_per_test, winsorize_pair_outcomes


def _raw_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clickability_test_id": ["t1", "t1", "t2", "t2", "t3"],
            "eyecatcher_id": ["e1", "e1", "e2", "e2", "e3"],
            "created_at": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"],
            "headline": ["A headline", "B headline", "Same", "Same", "Only one"],
            "excerpt": ["ex"] * 5,
            "lede": ["lede"] * 5,
            "share_text": ["share"] * 5,
            "impressions": [100, 100, 100, 200, 50],
            "clicks": [10, 20, 5, 8, 1],
            "source_split": ["unit"] * 5,
        }
    )


def test_clean_arm_level_keeps_only_multi_headline_tests():
    arms = clean_arm_level(_raw_fixture())
    assert arms["test_id"].nunique() == 1
    assert len(arms) == 2
    assert set(arms["headline"]) == {"A headline", "B headline"}
    assert np.isclose(arms.loc[arms["headline"] == "A headline", "ctr"].iloc[0], 0.10)


def test_clean_arm_level_can_filter_low_impression_arms():
    raw = _raw_fixture()
    raw.loc[raw["headline"] == "A headline", "impressions"] = 20
    arms = clean_arm_level(raw, min_arm_impressions=50)
    assert arms.empty


def test_make_one_pair_per_test_outputs_signed_regression_target():
    arms = clean_arm_level(_raw_fixture())
    pairs = make_one_pair_per_test(arms, seed=7, h_scale_size=1, target_size=1)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["headline_a"] != row["headline_b"]
    assert np.isclose(row["y_logit_ctr_diff"], row["logit_ctr_a"] - row["logit_ctr_b"])
    assert np.isclose(row["y_logit_ctr_diff_raw"], row["y_logit_ctr_diff"])
    assert np.isclose(row["y_ctr_diff_raw"], row["y_ctr_diff"])
    assert set(pairs["split"]) == {"h_scale"}


def test_winsorize_pair_outcomes_preserves_raw_values():
    pairs = pd.DataFrame(
        {
            "y_logit_ctr_diff": [-10.0, 0.0, 1.0, 10.0],
            "y_ctr_diff": [-0.2, 0.0, 0.01, 0.2],
        }
    )
    out = winsorize_pair_outcomes(pairs, logit_quantile=0.25, ctr_quantile=0.25)
    assert out["y_logit_ctr_diff"].min() > pairs["y_logit_ctr_diff"].min()
    assert out["y_logit_ctr_diff"].max() < pairs["y_logit_ctr_diff"].max()
    assert np.allclose(out["y_logit_ctr_diff_raw"], pairs["y_logit_ctr_diff"])
    assert int(out["y_logit_ctr_diff_winsorized"].sum()) == 2
