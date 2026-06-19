from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.upworthy_text_features import add_pair_differences, construct_arm_features, raw_headline_features


def test_raw_headline_features_detect_basic_cues():
    feats = raw_headline_features("Can You Name 3 Reasons This Is So Easy?")
    assert feats["word_count"] == 8
    assert feats["number_count"] == 1
    assert feats["has_numeric"] == 1
    assert feats["question_mark_count"] == 1
    assert feats["has_question"] == 1
    assert feats["interrogative_count"] >= 1
    assert feats["avg_word_length"] > 0
    assert feats["common_word_score"] > 0


def test_construct_arm_features_contains_requested_constructs():
    arms = pd.DataFrame(
        {
            "test_id": [1, 1, 2],
            "arm_id": [0, 1, 0],
            "headline": [
                "Can You Name 3 Reasons This Works?",
                "A Simple Story About People",
                "Extraordinary International Consequences",
            ],
        }
    )
    out = construct_arm_features(arms)
    for col in ["QUESTION", "NUMERIC", "SIMPLICITY", "COMMON", "LENGTH"]:
        assert col in out
        assert np.isfinite(out[col]).all()
    assert out.loc[0, "QUESTION"] == 1
    assert out.loc[0, "NUMERIC"] == 1


def test_add_pair_differences_uses_a_minus_b():
    arms = pd.DataFrame(
        {
            "test_id": [1, 1],
            "arm_id": [0, 1],
            "headline": ["Can You Name 3 Reasons This Works?", "A Simple Story About People"],
        }
    )
    arm_features = construct_arm_features(arms)
    pairs = pd.DataFrame({"pair_id": [0], "test_id": [1], "arm_id_a": [0], "arm_id_b": [1]})
    out = add_pair_differences(pairs, arm_features)
    for feature in ["QUESTION", "NUMERIC", "SIMPLICITY", "COMMON", "LENGTH"]:
        raw_delta = out[f"{feature}_a"].iloc[0] - out[f"{feature}_b"].iloc[0]
        assert np.isclose(out[f"delta_{feature}_raw"].iloc[0], raw_delta)
        assert np.isfinite(out[f"delta_{feature}"].iloc[0])
    for feature in ["QUESTION", "NUMERIC"]:
        assert np.isclose(out[f"delta_{feature}"].iloc[0], out[f"delta_{feature}_raw"].iloc[0])
