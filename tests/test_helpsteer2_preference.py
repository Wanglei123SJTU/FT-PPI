from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.prepare_helpsteer2_preference import (
    build_helpsteer2_preference_frame,
    has_structured_format,
    prompt_coverage,
)
from src.experiments.helpsteer2_embedding_extraction import pairwise_difference
from src.experiments.helpsteer2_embedding_mlp_scaling import antisymmetric_from_scores
from src.experiments.helpsteer2_lora_scaling import (
    DEFAULT_METHODS,
    build_pair_texts,
    compute_ols_and_if_weights,
    limit_indices,
    make_cell_plan,
    normalize_helpsteer_features,
    plan_for_worker,
    _load_or_compute_target_static,
    _load_or_tokenize_texts,
)


def _workspace_test_cache(name: str) -> Path:
    path = Path("artifacts") / "test_tmp" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def test_lora_pair_texts_include_forward_and_swapped_candidates():
    raw = pd.DataFrame(
        [
            {
                "prompt": "Explain churn",
                "response_1": "Short answer.",
                "response_2": "Detailed answer.",
            }
        ]
    )
    forward, swapped = build_pair_texts(raw)
    assert "Candidate A:\nShort answer." in forward[0]
    assert "Candidate B:\nDetailed answer." in forward[0]
    assert "Candidate A:\nDetailed answer." in swapped[0]
    assert "Candidate B:\nShort answer." in swapped[0]


def test_two_feature_if_weights_are_finite():
    raw = pd.DataFrame(
        {
            "y_preference_strength": [-1.0, 0.0, 1.0, 2.0],
            "delta_log_length": [-0.8, -0.2, 0.3, 1.1],
            "delta_log_sentences": [-0.6, 0.1, 0.4, 0.9],
        }
    )
    frame = normalize_helpsteer_features(raw)
    beta, hessian, if_weights = compute_ols_and_if_weights(
        frame,
        target="delta_log_length_scale",
        feature_columns=["delta_log_length_scale", "delta_log_sentences_scale"],
    )
    assert beta.shape == (3,)
    assert hessian.shape == (3, 3)
    assert if_weights.shape == (4,)
    assert np.isfinite(if_weights).all()


def test_lora_cell_plan_counts_targets_methods_s_and_reps():
    plan = make_cell_plan(
        targets=["delta_log_length_scale", "delta_log_sentences_scale"],
        methods=DEFAULT_METHODS[:2],
        s_grid=[50, 100],
        replications=3,
    )
    assert len(plan) == 2 * 2 * 2 * 3
    assert plan["task_index"].tolist() == list(range(len(plan)))
    assert set(plan["method"]) == {"mse_stop_mse", "mse_stop_ifvar"}


def test_limit_indices_is_deterministic_and_respects_limit():
    indices = np.arange(20)
    limited = limit_indices(indices, limit=5, seed=123)
    repeated = limit_indices(indices, limit=5, seed=123)
    assert len(limited) == 5
    assert np.array_equal(limited, repeated)
    assert np.all(np.diff(limited) > 0)
    assert np.array_equal(limit_indices(indices, limit=0, seed=123), indices)
    assert np.array_equal(limit_indices(indices, limit=99, seed=123), indices)


def test_plan_for_worker_partitions_cells_without_overlap():
    plan = make_cell_plan(targets=["delta_log_length_scale"], methods=DEFAULT_METHODS, s_grid=[50, 100], replications=2)
    shards = [plan_for_worker(plan, worker_index=i, num_workers=4) for i in range(4)]
    combined = sorted(int(task) for shard in shards for task in shard["task_index"])
    assert combined == plan["task_index"].tolist()
    assert sum(len(shard) for shard in shards) == len(plan)


def test_tokenization_cache_reuses_saved_tensors():
    torch = __import__("pytest").importorskip("torch")
    cache_dir = _workspace_test_cache("tokenization_cache")

    class CountingTokenizer:
        def __init__(self):
            self.calls = 0

        def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
            self.calls += 1
            assert padding is True
            assert truncation is True
            assert return_tensors == "pt"
            width = min(max_length, 4)
            values = torch.arange(len(texts) * width, dtype=torch.long).reshape(len(texts), width)
            return {"input_ids": values, "attention_mask": torch.ones_like(values)}

    tokenizer = CountingTokenizer()
    texts = ["first response", "second response"]
    first = _load_or_tokenize_texts(
        tokenizer,
        texts,
        max_length=8,
        cache_dir=cache_dir,
        model_name="toy/model",
        direction="forward",
    )
    second = _load_or_tokenize_texts(
        tokenizer,
        texts,
        max_length=8,
        cache_dir=cache_dir,
        model_name="toy/model",
        direction="forward",
    )
    assert tokenizer.calls == 1
    assert torch.equal(first["input_ids"], second["input_ids"])
    assert len(list(cache_dir.glob("tokenized_forward_*.pt"))) == 1
    shutil.rmtree(cache_dir)


def test_static_cache_reuses_if_weights():
    cache_dir = _workspace_test_cache("static_cache")
    raw = pd.DataFrame(
        {
            "y_preference_strength": [-1.0, 0.0, 1.0, 2.0, 1.5],
            "delta_log_length": [-0.8, -0.2, 0.3, 1.1, 0.6],
            "delta_log_sentences": [-0.6, 0.1, 0.4, 0.9, 0.5],
        }
    )
    frame = normalize_helpsteer_features(raw)
    kwargs = {
        "frame": frame,
        "target": "delta_log_length_scale",
        "feature_columns": ["delta_log_length_scale", "delta_log_sentences_scale"],
        "hessian_ridge": 0.0,
        "cache_dir": cache_dir,
    }
    first = _load_or_compute_target_static(**kwargs)
    second = _load_or_compute_target_static(**kwargs)
    assert first[:3] == second[:3]
    assert np.allclose(first[3], second[3])
    assert len(list(cache_dir.glob("static_delta_log_length_scale_*.npz"))) == 1
    shutil.rmtree(cache_dir)
