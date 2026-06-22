from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.upworthy_embedding_mlp_screening import (
    budget_rows_for_prediction,
    pair_embeddings,
)
from src.experiments.upworthy_openai_embeddings import choose_screening_sample, ordered_unique_texts


def test_choose_screening_sample_uses_h_scale_train_pool_and_target_eval():
    frame = pd.DataFrame(
        {
            "split": ["h_scale"] * 6 + ["target"] * 4,
            "pair_id": list(range(10)),
            "headline_a": [f"a{i}" for i in range(10)],
            "headline_b": [f"b{i}" for i in range(10)],
            "y_logit_ctr_diff": np.arange(10, dtype=float),
        }
    )
    sample = choose_screening_sample(frame, n_pairs=6, n_eval=2, seed=1)
    assert len(sample) == 6
    assert (sample["screen_split"] == "train_pool").sum() == 4
    assert (sample["screen_split"] == "evaluation").sum() == 2
    assert set(sample.loc[sample["screen_split"] == "train_pool", "split"]) == {"h_scale"}
    assert set(sample.loc[sample["screen_split"] == "evaluation", "split"]) == {"target"}


def test_ordered_unique_texts_deduplicates_across_arms():
    frame = pd.DataFrame({"headline_a": ["same", "left"], "headline_b": ["right", "same"]})
    assert ordered_unique_texts(frame) == ["same", "left", "right"]


def test_pair_embedding_diff_is_antisymmetric_under_swapping():
    cache = {
        "embedding_a": np.array([[1.0, 2.0], [5.0, 1.0]], dtype=np.float32),
        "embedding_b": np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float32),
    }
    swapped = {"embedding_a": cache["embedding_b"], "embedding_b": cache["embedding_a"]}
    assert np.allclose(pair_embeddings(swapped, "diff"), -pair_embeddings(cache, "diff"))


def test_budget_ratio_requires_variance_gain_to_offset_spent_training_labels():
    rows = budget_rows_for_prediction(
        target="delta_test",
        method="m",
        s=200,
        ifvar=0.6,
        direct_ols_ifvar=1.0,
        zero_ifvar=1.2,
        budgets=[500, 1000],
    )
    by_budget = {row["budget"]: row for row in rows}
    assert by_budget[500]["ratio_vs_direct_ols"] == 1.0
    assert by_budget[500]["beats_direct_ols"] is False
    assert by_budget[1000]["ratio_vs_direct_ols"] == 0.75
    assert by_budget[1000]["beats_direct_ols"] is True
