from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.upworthy_feature_screening import (
    build_candidate_features,
    ess_share,
    _scale_no_center,
)


def test_scale_no_center_preserves_signed_symmetry():
    values = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
    scaled = _scale_no_center(values)
    assert np.isclose(float(scaled.mean()), 0.0)
    assert np.isclose(float(scaled.iloc[0]), -float(scaled.iloc[-1]))
    assert np.isclose(float(scaled.iloc[1]), -float(scaled.iloc[-2]))


def test_if_ess_share_penalizes_concentrated_weights():
    uniform = np.ones(10)
    concentrated = np.array([10.0] + [0.1] * 9)
    assert np.isclose(ess_share(uniform), 1.0)
    assert ess_share(concentrated) < 0.2


def test_candidate_features_include_context_and_sentiment_deltas():
    frame = pd.DataFrame(
        {
            "headline_a": ["Why This Simple Trick Works", "Cats Save A Town"],
            "headline_b": ["A Simple Trick Works", "A Town Was Saved"],
            "excerpt_a": ["This simple trick works for readers", "A town was saved by cats"],
            "excerpt_b": ["This simple trick works for readers", "A town was saved by cats"],
            "lede_a": ["More context here", "More context here"],
            "lede_b": ["More context here", "More context here"],
            "share_text_a": ["share", "share"],
            "share_text_b": ["share", "share"],
            "split": ["target", "target"],
            "y_logit_ctr_diff": [0.1, -0.2],
        }
    )
    out, metadata = build_candidate_features(frame)
    assert "delta_context_coverage" in out.columns
    assert "delta_vader_compound" in out.columns
    assert "delta_curiosity_style" in out.columns
    families = set(metadata["family"])
    assert "context_alignment" in families
    assert "sentiment" in families
