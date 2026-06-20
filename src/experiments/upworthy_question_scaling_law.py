from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from src.experiments.wine_var_scaling_law import fit_scaling_law, ids_hash, is_cuda_oom, write_json

try:
    from src.data.upworthy_text_features import FEATURES as UPWORTHY_TEXT_FEATURES
except Exception:  # pragma: no cover - keep CLI usable if optional feature deps are missing
    UPWORTHY_TEXT_FEATURES = []

Y_COL = "y_logit_ctr_diff"
TEXT_A_COL = "headline_a"
TEXT_B_COL = "headline_b"
ID_COL = "sample_id"
EST_WEIGHT_COL = "estimation_weight"
FEATURE_COLS = [
    "delta_QUESTION",
    "delta_NUMERIC",
    "delta_SIMPLICITY",
    "delta_COMMON",
    "delta_LENGTH",
]
KNOWN_FEATURE_COLS = sorted(set(FEATURE_COLS) | {f"delta_{feature}" for feature in UPWORTHY_TEXT_FEATURES})
TARGET_FEATURE = "delta_QUESTION"
IF_WEIGHT_COL = "if_weight_target"
LEGACY_IF_WEIGHT_COL = "if_weight_question"


@dataclass(frozen=True)
class PopulationSplit:
    h_scale_ids: np.ndarray
    p_target_ids: np.ndarray


@dataclass(frozen=True)
class ScalingSplit:
    train_pool_ids: np.ndarray
    v_stop_ids: np.ndarray
    v_scale_ids: np.ndarray
    train_order_ids: np.ndarray


@dataclass(frozen=True)
class MethodSpec:
    name: str
    loss: str
    early_stopping_metric: str
    sampling_strategy: str = "uniform"
    warmup_loss: str | None = None
    warmup_epochs: int = 0
    if_weight_clip: float | None = None
    train_backbone: bool = True
    lora_lr: float | None = None
    head_lr: float | None = None
    batch_size: int | None = None
    max_epochs: int | None = None
    min_epochs: int | None = None
    early_stopping_patience: int | None = None


@dataclass(frozen=True)
class OutcomeScale:
    mean: float
    sd: float


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def outcome_col_from_config(config: dict[str, Any] | None = None) -> str:
    if config is None:
        return Y_COL
    return str(config.get("outcome_column", Y_COL))


def outcome_transform_from_config(config: dict[str, Any] | None = None) -> str:
    if config is None:
        return ""
    return str(config.get("outcome_transform", "")).lower()


def estimation_weight_scheme_from_config(config: dict[str, Any] | None = None) -> str:
    if config is None:
        return "none"
    return str(config.get("estimation_weight_scheme", config.get("estimation_weights", "none"))).lower()


def _require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{context} requires missing columns: {missing}")


def build_estimation_weights(df: pd.DataFrame, scheme: str) -> np.ndarray:
    value = str(scheme).lower()
    if value in {"", "none", "unweighted", "uniform"}:
        return np.ones(len(df), dtype=float)
    if value in {"logit_precision", "logit_precision_p99clip"}:
        _require_columns(df, ["clicks_a", "clicks_b", "impressions_a", "impressions_b"], value)
        clicks_a = df["clicks_a"].astype(float).to_numpy()
        clicks_b = df["clicks_b"].astype(float).to_numpy()
        imps_a = df["impressions_a"].astype(float).to_numpy()
        imps_b = df["impressions_b"].astype(float).to_numpy()
        var = (
            1.0 / (clicks_a + 0.5)
            + 1.0 / (np.maximum(imps_a - clicks_a, 0.0) + 0.5)
            + 1.0 / (clicks_b + 0.5)
            + 1.0 / (np.maximum(imps_b - clicks_b, 0.0) + 0.5)
        )
        weights = 1.0 / np.maximum(var, 1e-12)
        if value.endswith("p99clip"):
            weights = np.minimum(weights, float(np.quantile(weights, 0.99)))
        return weights.astype(float)
    if value == "impression_hmean":
        _require_columns(df, ["impressions_a", "impressions_b"], value)
        imps_a = df["impressions_a"].astype(float).to_numpy()
        imps_b = df["impressions_b"].astype(float).to_numpy()
        return (1.0 / np.maximum(1.0 / np.maximum(imps_a, 1e-12) + 1.0 / np.maximum(imps_b, 1e-12), 1e-12)).astype(float)
    raise ValueError(f"unsupported estimation_weight_scheme {scheme!r}")


def apply_row_filters(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    if config is None:
        return df
    out = df
    if config.get("min_impressions_per_arm") not in (None, ""):
        _require_columns(out, ["impressions_a", "impressions_b"], "min_impressions_per_arm")
        threshold = float(config["min_impressions_per_arm"])
        out = out.loc[(out["impressions_a"].astype(float) >= threshold) & (out["impressions_b"].astype(float) >= threshold)]
    if config.get("min_total_impressions") not in (None, ""):
        _require_columns(out, ["impressions_a", "impressions_b"], "min_total_impressions")
        threshold = float(config["min_total_impressions"])
        out = out.loc[(out["impressions_a"].astype(float) + out["impressions_b"].astype(float)) >= threshold]
    if config.get("min_clicks_per_arm") not in (None, ""):
        _require_columns(out, ["clicks_a", "clicks_b"], "min_clicks_per_arm")
        threshold = float(config["min_clicks_per_arm"])
        out = out.loc[(out["clicks_a"].astype(float) >= threshold) & (out["clicks_b"].astype(float) >= threshold)]
    return out.reset_index(drop=True)


def _safe_logit(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob.astype(float), 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def compute_transformed_outcome(df: pd.DataFrame, transform: str, config: dict[str, Any] | None = None) -> np.ndarray:
    value = str(transform).lower()
    if value in {"", "none"}:
        raise ValueError("compute_transformed_outcome requires a non-empty outcome_transform")
    if value in {"logit_ctr_diff_eb", "ctr_diff_eb"}:
        _require_columns(df, ["clicks_a", "clicks_b", "impressions_a", "impressions_b"], value)
        tau = float((config or {}).get("ctr_shrinkage_tau", 0.0))
        if tau < 0:
            raise ValueError("ctr_shrinkage_tau must be non-negative")
        clicks_a = df["clicks_a"].astype(float).to_numpy()
        clicks_b = df["clicks_b"].astype(float).to_numpy()
        imps_a = df["impressions_a"].astype(float).to_numpy()
        imps_b = df["impressions_b"].astype(float).to_numpy()
        total_imps = float(imps_a.sum() + imps_b.sum())
        if total_imps <= 0:
            raise ValueError("CTR outcome transform requires positive total impressions")
        prior_ctr = float((clicks_a.sum() + clicks_b.sum()) / total_imps)
        ctr_a = (clicks_a + tau * prior_ctr) / np.maximum(imps_a + tau, 1e-12)
        ctr_b = (clicks_b + tau * prior_ctr) / np.maximum(imps_b + tau, 1e-12)
        if value == "ctr_diff_eb":
            return (ctr_a - ctr_b).astype(float)
        return (_safe_logit(ctr_a) - _safe_logit(ctr_b)).astype(float)
    raise ValueError(f"unsupported outcome_transform {transform!r}")


def load_upworthy_pairs(input_csv: str | Path, config: dict[str, Any] | None = None) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    outcome_col = outcome_col_from_config(config)
    outcome_transform = outcome_transform_from_config(config)
    feature_cols = feature_cols_from_config(config)
    required = {TEXT_A_COL, TEXT_B_COL, "split", *feature_cols}
    if outcome_transform in {"", "none"}:
        required.add(outcome_col)
    elif outcome_transform in {"logit_ctr_diff_eb", "ctr_diff_eb"}:
        required.update(["clicks_a", "clicks_b", "impressions_a", "impressions_b"])
    else:
        raise ValueError(f"unsupported outcome_transform {outcome_transform!r}")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    out = apply_row_filters(df.copy(), config)
    if outcome_transform not in {"", "none"}:
        out[outcome_col] = compute_transformed_outcome(out, outcome_transform, config)
    if ID_COL not in out.columns:
        if "pair_id" in out.columns:
            out.insert(0, ID_COL, out["pair_id"].astype(np.int64))
        else:
            out.insert(0, ID_COL, np.arange(len(out), dtype=np.int64))
    out[ID_COL] = out[ID_COL].astype(np.int64)
    if out[ID_COL].duplicated().any():
        raise ValueError(f"{ID_COL} must be unique")
    out[Y_COL] = out[outcome_col].astype(float)
    out[EST_WEIGHT_COL] = build_estimation_weights(out, estimation_weight_scheme_from_config(config))
    for col in feature_cols:
        out[col] = out[col].astype(float)
    out[TEXT_A_COL] = out[TEXT_A_COL].astype(str)
    out[TEXT_B_COL] = out[TEXT_B_COL].astype(str)
    return out.reset_index(drop=True)


def feature_cols_from_config(config: dict[str, Any] | None = None) -> list[str]:
    if config is None or config.get("feature_columns") in (None, ""):
        return list(FEATURE_COLS)
    cols = [str(col) for col in config["feature_columns"]]
    if not cols:
        raise ValueError("feature_columns must contain at least one feature")
    unknown = sorted(set(cols) - set(KNOWN_FEATURE_COLS))
    if unknown:
        raise ValueError(f"feature_columns must be drawn from known Upworthy feature columns, got unknown {unknown}")
    if len(cols) != len(set(cols)):
        raise ValueError(f"feature_columns contains duplicates: {cols}")
    return cols


def surrogate_feature_cols_from_config(config: dict[str, Any] | None = None) -> list[str]:
    if config is None or "surrogate_feature_columns" not in config or config.get("surrogate_feature_columns") is None:
        return feature_cols_from_config(config)
    raw = config.get("surrogate_feature_columns")
    if isinstance(raw, str):
        if raw.lower() in {"", "none", "text_only", "text-only"}:
            return []
        raise ValueError("surrogate_feature_columns must be a list of feature names or [] for text-only")
    cols = [str(col) for col in raw]
    unknown = sorted(set(cols) - set(KNOWN_FEATURE_COLS))
    if unknown:
        raise ValueError(f"surrogate_feature_columns must be drawn from known Upworthy feature columns, got unknown {unknown}")
    if len(cols) != len(set(cols)):
        raise ValueError(f"surrogate_feature_columns contains duplicates: {cols}")
    return cols


def normalize_sampling_strategy(strategy: str) -> str:
    value = str(strategy).lower()
    aliases = {
        "question_balanced": "target_nonzero_balanced",
        "question_if_weight_balanced": "target_if_weight_balanced",
    }
    return aliases.get(value, value)


def build_population_split(df: pd.DataFrame) -> PopulationSplit:
    h_scale = df.loc[df["split"].astype(str) == "h_scale", ID_COL].to_numpy(dtype=np.int64)
    p_target = df.loc[df["split"].astype(str) == "target", ID_COL].to_numpy(dtype=np.int64)
    if len(h_scale) == 0 or len(p_target) == 0:
        raise ValueError("input data must contain split values 'h_scale' and 'target'")
    return PopulationSplit(h_scale_ids=h_scale, p_target_ids=p_target)


def method_specs(config: dict[str, Any]) -> list[MethodSpec]:
    raw = config.get("methods", [{"name": "mse_stop_mse", "loss": "mse", "early_stopping_metric": "mse"}])
    methods: list[MethodSpec] = []
    seen: set[str] = set()
    allowed_losses = {"mse", "weighted_mse", "ifvarq"}
    allowed_stop_metrics = {"mse", "ifvarq"}
    allowed_sampling = {
        "uniform",
        "if_weight_balanced",
        "target_nonzero_balanced",
        "target_if_weight_balanced",
        "question_balanced",
        "question_if_weight_balanced",
    }
    for item in raw:
        name = str(item["name"]).lower()
        loss = str(item.get("loss", name)).lower()
        stop = str(item.get("early_stopping_metric", loss)).lower()
        warmup_loss = item.get("warmup_loss")
        warmup_loss = None if warmup_loss in (None, "") else str(warmup_loss).lower()
        sampling_strategy = str(item.get("sampling_strategy", config.get("sampling_strategy", "uniform"))).lower()
        if loss not in allowed_losses:
            raise ValueError(f"unsupported loss {loss!r}; expected one of {sorted(allowed_losses)}")
        if stop not in allowed_stop_metrics:
            raise ValueError(f"unsupported early_stopping_metric {stop!r}; expected 'mse' or 'ifvarq'")
        if warmup_loss is not None and warmup_loss not in allowed_losses:
            raise ValueError(f"unsupported warmup_loss {warmup_loss!r}; expected one of {sorted(allowed_losses)}")
        if sampling_strategy not in allowed_sampling:
            raise ValueError(f"unsupported sampling_strategy {sampling_strategy!r}; expected one of {sorted(allowed_sampling)}")
        if name in seen:
            raise ValueError(f"duplicate method name {name!r}")
        seen.add(name)
        if_weight_clip = item.get("if_weight_clip", config.get("if_weight_clip"))
        methods.append(
            MethodSpec(
                name=name,
                loss=loss,
                early_stopping_metric=stop,
                sampling_strategy=sampling_strategy,
                warmup_loss=warmup_loss,
                warmup_epochs=int(item.get("warmup_epochs", 0)),
                if_weight_clip=None if if_weight_clip in (None, "") else float(if_weight_clip),
                train_backbone=bool(item.get("train_backbone", config.get("train_backbone", True))),
                lora_lr=None if item.get("lora_lr") in (None, "") else float(item["lora_lr"]),
                head_lr=None if item.get("head_lr") in (None, "") else float(item["head_lr"]),
                batch_size=None if item.get("batch_size") in (None, "") else int(item["batch_size"]),
                max_epochs=None if item.get("max_epochs") in (None, "") else int(item["max_epochs"]),
                min_epochs=None if item.get("min_epochs") in (None, "") else int(item["min_epochs"]),
                early_stopping_patience=None
                if item.get("early_stopping_patience") in (None, "")
                else int(item["early_stopping_patience"]),
            )
        )
    return methods


def method_by_name(config: dict[str, Any], name: str) -> MethodSpec:
    methods = {method.name: method for method in method_specs(config)}
    key = str(name).lower()
    if key not in methods:
        raise ValueError(f"unknown method {name!r}; expected one of {sorted(methods)}")
    return methods[key]


def rep_seed(config: dict[str, Any], replication_id: int, salt: int = 0) -> int:
    return int(config.get("seed", 20260618)) + 100_003 * int(replication_id) + int(salt)


def _interleave_permuted_groups(groups: list[np.ndarray], cycle: list[int], rng: np.random.Generator) -> np.ndarray:
    shuffled = [rng.permutation(np.asarray(group, dtype=np.int64)) for group in groups]
    positions = [0 for _ in shuffled]
    order: list[int] = []
    total = int(sum(len(group) for group in shuffled))
    if total == 0:
        return np.asarray([], dtype=np.int64)
    while len(order) < total:
        advanced = False
        for group_index in cycle:
            if len(order) >= total:
                break
            if positions[group_index] < len(shuffled[group_index]):
                order.append(int(shuffled[group_index][positions[group_index]]))
                positions[group_index] += 1
                advanced = True
        if not advanced:
            leftovers = [
                int(group[pos])
                for group, start in zip(shuffled, positions)
                for pos in range(start, len(group))
            ]
            order.extend(rng.permutation(np.asarray(leftovers, dtype=np.int64)).tolist())
    return np.asarray(order, dtype=np.int64)


def build_train_order(
    df: pd.DataFrame,
    train_pool_ids: np.ndarray,
    rng: np.random.Generator,
    sampling_strategy: str,
    target_feature: str = TARGET_FEATURE,
) -> np.ndarray:
    strategy = normalize_sampling_strategy(sampling_strategy)
    if strategy == "uniform":
        return rng.permutation(np.asarray(train_pool_ids, dtype=np.int64))

    weight_col = IF_WEIGHT_COL if IF_WEIGHT_COL in df.columns else LEGACY_IF_WEIGHT_COL
    pool = pd.DataFrame({ID_COL: np.asarray(train_pool_ids, dtype=np.int64)}).merge(
        df[[ID_COL, target_feature, weight_col]],
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )
    if pool[target_feature].isna().any() or pool[weight_col].isna().any():
        raise ValueError("train pool contains unknown sample ids or missing influence weights")

    ids = pool[ID_COL].to_numpy(dtype=np.int64)

    nonzero = pool.loc[pool[target_feature].abs() > 0, ID_COL].to_numpy(dtype=np.int64)
    zero = pool.loc[pool[target_feature].abs() == 0, ID_COL].to_numpy(dtype=np.int64)
    abs_weight = pool[weight_col].abs().to_numpy(dtype=float)
    cutoff = float(np.quantile(abs_weight, 0.75))
    high_weight = pool.loc[pool[weight_col].abs() >= cutoff, ID_COL].to_numpy(dtype=np.int64)
    low_weight = pool.loc[pool[weight_col].abs() < cutoff, ID_COL].to_numpy(dtype=np.int64)

    if strategy == "target_nonzero_balanced":
        if len(zero) == 0:
            return rng.permutation(ids)
        return _interleave_permuted_groups([nonzero, zero], [0, 1], rng)
    if strategy == "if_weight_balanced":
        return _interleave_permuted_groups([high_weight, low_weight], [0, 1], rng)
    if strategy == "target_if_weight_balanced":
        nonzero_ids = set(map(int, nonzero))
        high_ids = set(map(int, high_weight))
        groups = [
            pool.loc[pool[ID_COL].isin(nonzero_ids & high_ids), ID_COL].to_numpy(dtype=np.int64),
            pool.loc[pool[ID_COL].isin(nonzero_ids - high_ids), ID_COL].to_numpy(dtype=np.int64),
            pool.loc[pool[ID_COL].isin(high_ids - nonzero_ids), ID_COL].to_numpy(dtype=np.int64),
            pool.loc[~pool[ID_COL].isin(nonzero_ids | high_ids), ID_COL].to_numpy(dtype=np.int64),
        ]
        return _interleave_permuted_groups(groups, [0, 1, 2, 0, 1, 3], rng)

    raise ValueError(f"unsupported sampling_strategy {sampling_strategy!r}")


def build_scaling_split(
    df: pd.DataFrame,
    config: dict[str, Any],
    replication_id: int,
    sampling_strategy: str = "uniform",
    target_feature: str = TARGET_FEATURE,
) -> ScalingSplit:
    population = build_population_split(df)
    train_pool_size = int(config["train_pool_size"])
    v_stop_size = int(config["validation_stop_size"])
    v_scale_size = int(config["validation_scale_size"])
    required = train_pool_size + v_stop_size + v_scale_size
    if required > len(population.h_scale_ids):
        raise ValueError(f"need {required} H_scale rows, found {len(population.h_scale_ids)}")

    rng = np.random.default_rng(rep_seed(config, replication_id))
    ordered = rng.permutation(population.h_scale_ids)
    v_stop_ids = np.asarray(ordered[:v_stop_size], dtype=np.int64)
    v_scale_ids = np.asarray(ordered[v_stop_size : v_stop_size + v_scale_size], dtype=np.int64)
    train_pool_ids = np.asarray(ordered[v_stop_size + v_scale_size : required], dtype=np.int64)
    train_order_ids = build_train_order(df, train_pool_ids, rng, sampling_strategy, target_feature=target_feature)
    split = ScalingSplit(
        train_pool_ids=train_pool_ids,
        v_stop_ids=v_stop_ids,
        v_scale_ids=v_scale_ids,
        train_order_ids=np.asarray(train_order_ids, dtype=np.int64),
    )
    validate_scaling_split(split, config)
    return split


def validate_scaling_split(split: ScalingSplit, config: dict[str, Any]) -> None:
    train_pool = set(map(int, split.train_pool_ids))
    v_stop = set(map(int, split.v_stop_ids))
    v_scale = set(map(int, split.v_scale_ids))
    if len(train_pool) != len(split.train_pool_ids):
        raise ValueError("train pool has duplicate sample ids")
    if len(v_stop) != len(split.v_stop_ids):
        raise ValueError("validation stop split has duplicate sample ids")
    if len(v_scale) != len(split.v_scale_ids):
        raise ValueError("validation scale split has duplicate sample ids")
    if train_pool & v_stop or train_pool & v_scale or v_stop & v_scale:
        raise ValueError("train/stop/scale splits must be disjoint")
    previous: set[int] = set()
    for s_train in s_grid(config):
        current = set(map(int, train_ids_for_s(split, s_train)))
        if not current.issubset(train_pool):
            raise ValueError(f"train set s={s_train} is not a subset of train pool")
        if not previous.issubset(current):
            raise ValueError("nested train sets are not monotone")
        previous = current


def train_ids_for_s(split: ScalingSplit, s_train: int) -> np.ndarray:
    if int(s_train) > len(split.train_order_ids):
        raise ValueError(f"s_train={s_train} exceeds train pool size={len(split.train_order_ids)}")
    return split.train_order_ids[: int(s_train)]


def s_grid(config: dict[str, Any]) -> list[int]:
    return [int(x) for x in config["s_grid"]]


def replication_ids(config: dict[str, Any]) -> list[int]:
    return [int(x) for x in config["replication_ids"]]


def x_matrix(frame: pd.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    return np.column_stack([np.ones(len(frame), dtype=float), frame[cols].to_numpy(dtype=float)])


def target_feature_index(target_feature: str = TARGET_FEATURE, feature_cols: list[str] | None = None) -> int:
    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    if target_feature not in cols:
        raise ValueError(f"target_feature must be one of active feature_columns={cols}, got {target_feature!r}")
    return 1 + cols.index(target_feature)


def fit_ols_beta(frame: pd.DataFrame, feature_cols: list[str] | None = None, weight_col: str | None = None) -> np.ndarray:
    x = x_matrix(frame, feature_cols)
    y = frame[Y_COL].to_numpy(dtype=float)
    if weight_col is None:
        return np.linalg.pinv(x.T @ x) @ x.T @ y
    weights = frame[weight_col].to_numpy(dtype=float)
    return np.linalg.pinv(x.T * weights @ x) @ (x.T * weights @ y)


def compute_inference_setup(
    df: pd.DataFrame,
    p_target_ids: Iterable[int],
    y_scale: OutcomeScale,
    target_feature: str = TARGET_FEATURE,
    feature_cols: list[str] | None = None,
    hessian_ridge: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    ridge = float(hessian_ridge)
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("hessian_ridge must be a nonnegative finite value")
    if EST_WEIGHT_COL not in df.columns:
        df = df.copy()
        df[EST_WEIGHT_COL] = 1.0
    target = subset_by_ids(df, np.asarray(list(p_target_ids), dtype=np.int64), "target_for_inference", y_scale)
    raw_target_weights = target[EST_WEIGHT_COL].to_numpy(dtype=float)
    target_weight_mean = float(np.mean(raw_target_weights))
    if not np.isfinite(target_weight_mean) or target_weight_mean <= 0:
        raise ValueError("target estimation weights must have positive finite mean")
    normalized_all_weights = df[EST_WEIGHT_COL].to_numpy(dtype=float) / target_weight_mean
    normalized_target_weights = raw_target_weights / target_weight_mean
    target = target.copy()
    target[EST_WEIGHT_COL] = normalized_target_weights
    x_target = x_matrix(target, cols)
    hessian = (x_target.T * normalized_target_weights @ x_target) / len(target)
    singular_values = np.linalg.svd(hessian, compute_uv=False)
    min_singular = float(np.min(singular_values))
    max_singular = float(np.max(singular_values))
    condition_number = float(max_singular / min_singular) if min_singular > 0 else float("inf")
    ridge_matrix = hessian + ridge * np.eye(hessian.shape[0], dtype=float)
    hessian_inv = np.linalg.pinv(ridge_matrix)
    idx = target_feature_index(target_feature, cols)
    beta = fit_ols_beta(target, cols, weight_col=EST_WEIGHT_COL)
    weights_all = (x_matrix(df, cols) @ hessian_inv[idx, :]) * normalized_all_weights
    out = df.copy()
    out[EST_WEIGHT_COL] = normalized_all_weights
    out[IF_WEIGHT_COL] = weights_all
    out[LEGACY_IF_WEIGHT_COL] = weights_all

    target_weights = (x_target @ hessian_inv[idx, :]) * normalized_target_weights
    target_residual_raw = target[Y_COL].to_numpy(dtype=float) - x_target @ beta
    target_if_residual_raw = target_weights * target_residual_raw
    target_if_residual_scaled = target_if_residual_raw / y_scale.sd
    names = ["intercept", *cols]
    info = {
        "target_feature": target_feature,
        "target_feature_index": int(idx),
        "feature_columns": names,
        "target_beta_raw": {name: float(value) for name, value in zip(names, beta)},
        "target_coefficient_raw": float(beta[idx]),
        "question_beta_raw": float(beta[idx]),
        "hessian": hessian.tolist(),
        "hessian_singular_values": [float(value) for value in singular_values],
        "hessian_condition_number": condition_number,
        "hessian_ridge": ridge,
        "hessian_inv": hessian_inv.tolist(),
        "if_weight_abs_quantiles": {
            str(q): float(value)
            for q, value in zip(
                [0.5, 0.9, 0.95, 0.99, 0.999, 1.0],
                np.quantile(np.abs(weights_all), [0.5, 0.9, 0.95, 0.99, 0.999, 1.0]),
            )
        },
        "direct_ols_ifvar_target_raw": float(np.var(target_if_residual_raw, ddof=1)),
        "direct_ols_ifvar_target_scaled": float(np.var(target_if_residual_scaled, ddof=1)),
        "estimation_weight_mean_raw_target": float(target_weight_mean),
        "estimation_weight_min_normalized_target": float(np.min(normalized_target_weights)),
        "estimation_weight_max_normalized_target": float(np.max(normalized_target_weights)),
        "target_size": int(len(target)),
    }
    return out, info


def outcome_scale_from_h_scale(df: pd.DataFrame, h_scale_ids: Iterable[int]) -> OutcomeScale:
    h_ids = np.asarray(list(h_scale_ids), dtype=np.int64)
    h_frame = df.loc[df[ID_COL].isin(h_ids)]
    mean = float(h_frame[Y_COL].mean())
    sd = float(h_frame[Y_COL].std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("H_scale outcome standard deviation must be positive")
    return OutcomeScale(mean=mean, sd=sd)


def scaled_y(y_raw: Any, y_scale: OutcomeScale) -> np.ndarray:
    return (np.asarray(y_raw, dtype=float) - y_scale.mean) / y_scale.sd


def raw_y(y_scaled: Any, y_scale: OutcomeScale) -> np.ndarray:
    return y_scale.mean + y_scale.sd * np.asarray(y_scaled, dtype=float)


def constant_prediction_frame(frame: pd.DataFrame, y_scale: OutcomeScale) -> pd.DataFrame:
    """Prediction frame for the no-text surrogate f(x)=E_Hscale[Y]."""
    weight_col = IF_WEIGHT_COL if IF_WEIGHT_COL in frame.columns else LEGACY_IF_WEIGHT_COL
    out = pd.DataFrame(
        {
            ID_COL: frame[ID_COL].astype(np.int64).to_numpy(),
            "y_raw": frame["y_raw"].astype(float).to_numpy(),
            "y_scaled": frame["y_scaled"].astype(float).to_numpy(),
            "pred_scaled": np.zeros(len(frame), dtype=float),
            "pred_raw": np.full(len(frame), y_scale.mean, dtype=float),
            EST_WEIGHT_COL: (
                frame[EST_WEIGHT_COL].astype(float).to_numpy()
                if EST_WEIGHT_COL in frame.columns
                else np.ones(len(frame), dtype=float)
            ),
            IF_WEIGHT_COL: frame[weight_col].astype(float).to_numpy(),
            LEGACY_IF_WEIGHT_COL: frame[weight_col].astype(float).to_numpy(),
        }
    )
    out["residual_scaled"] = out["y_scaled"] - out["pred_scaled"]
    out["residual_raw"] = out["y_raw"] - out["pred_raw"]
    out["if_residual_scaled"] = out[IF_WEIGHT_COL] * out["residual_scaled"]
    out["if_residual_raw"] = out[IF_WEIGHT_COL] * out["residual_raw"]
    return out


def subset_by_ids(df: pd.DataFrame, ids: np.ndarray, role: str, y_scale: OutcomeScale) -> pd.DataFrame:
    ids_df = pd.DataFrame({ID_COL: np.asarray(ids, dtype=np.int64), "_order": np.arange(len(ids))})
    out = ids_df.merge(df, on=ID_COL, how="left", validate="one_to_one").sort_values("_order")
    if out[Y_COL].isna().any():
        raise ValueError(f"{role} contains unknown sample ids")
    out = out.drop(columns=["_order"]).reset_index(drop=True)
    out["split_role"] = role
    out["y_raw"] = out[Y_COL].astype(float)
    out["y_scaled"] = scaled_y(out["y_raw"], y_scale)
    return out


def mse_loss_from_residuals(residual: Any) -> Any:
    return (residual**2).mean()


def weighted_mse_loss_from_residuals(residual: Any, sample_weight: Any) -> Any:
    mean_weight = sample_weight.mean()
    if hasattr(mean_weight, "clamp"):
        mean_weight = mean_weight.clamp(min=1e-12)
    else:
        mean_weight = max(float(mean_weight), 1e-12)
    weight = sample_weight / mean_weight
    return (weight * residual**2).mean()


def ifvarq_loss_from_residuals(residual: Any, influence_weight: Any, if_weight_clip: float | None = None) -> Any:
    if if_weight_clip is not None and float(if_weight_clip) > 0:
        influence_weight = influence_weight.clip(min=-float(if_weight_clip), max=float(if_weight_clip))
    if_residual = influence_weight * residual
    return ((if_residual - if_residual.mean()) ** 2).mean()


def active_loss_name(method: MethodSpec, epoch: int) -> str:
    if method.warmup_loss is not None and int(method.warmup_epochs) > 0 and int(epoch) <= int(method.warmup_epochs):
        return method.warmup_loss
    return method.loss


def training_loss_from_residuals(
    residual: Any,
    influence_weight: Any,
    method: MethodSpec | str,
    epoch: int = 1,
    sample_weight: Any | None = None,
) -> Any:
    if isinstance(method, MethodSpec):
        loss = active_loss_name(method, epoch)
        clip = method.if_weight_clip
    else:
        loss = str(method).lower()
        clip = None
    if loss == "mse":
        return mse_loss_from_residuals(residual)
    if loss == "weighted_mse":
        if sample_weight is None:
            raise ValueError("weighted_mse requires sample_weight")
        return weighted_mse_loss_from_residuals(residual, sample_weight)
    if loss == "ifvarq":
        return ifvarq_loss_from_residuals(residual, influence_weight, clip)
    raise ValueError(f"unsupported loss {loss!r}")


def stopping_value_from_metrics(metrics: dict[str, float], early_stopping_metric: str) -> float:
    metric = str(early_stopping_metric).lower()
    if metric == "mse":
        return float(metrics["rmse_scaled"]) ** 2
    if metric == "ifvarq":
        return float(metrics["if_residual_var_scaled"])
    raise ValueError(f"unsupported early stopping metric {early_stopping_metric!r}")


def steps_per_epoch_for_s(s_train: int, batch_size: int) -> int:
    return max(1, math.ceil(int(s_train) / int(batch_size)))


def _require_training_modules() -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("training requires torch, transformers, peft, accelerate, and pyyaml") from exc
    from torch.utils.data import DataLoader

    return {
        "torch": torch,
        "DataLoader": DataLoader,
        "AutoModel": AutoModel,
        "AutoTokenizer": AutoTokenizer,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
    }


def set_training_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


class PairTokenizedDataset:
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int, feature_cols: list[str] | None = None):
        cols = feature_cols if feature_cols is not None else FEATURE_COLS
        weight_col = IF_WEIGHT_COL if IF_WEIGHT_COL in frame.columns else LEGACY_IF_WEIGHT_COL
        self.sample_ids = frame[ID_COL].astype(int).tolist()
        self.y_raw = frame["y_raw"].astype(float).to_numpy()
        self.y_scaled = frame["y_scaled"].astype(float).to_numpy()
        self.influence_weight = frame[weight_col].astype(float).to_numpy()
        self.estimation_weight = frame[EST_WEIGHT_COL].astype(float).to_numpy() if EST_WEIGHT_COL in frame.columns else np.ones(len(frame), dtype=float)
        self.features = frame[cols].astype(float).to_numpy()
        self.a_encodings = tokenizer(
            frame[TEXT_A_COL].astype(str).tolist(),
            truncation=True,
            max_length=int(max_length),
            padding=False,
        )
        self.b_encodings = tokenizer(
            frame[TEXT_B_COL].astype(str).tolist(),
            truncation=True,
            max_length=int(max_length),
            padding=False,
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "sample_id": self.sample_ids[idx],
            "input_ids_a": self.a_encodings["input_ids"][idx],
            "attention_mask_a": self.a_encodings["attention_mask"][idx],
            "input_ids_b": self.b_encodings["input_ids"][idx],
            "attention_mask_b": self.b_encodings["attention_mask"][idx],
            "features": self.features[idx],
            "y_raw": float(self.y_raw[idx]),
            "y_scaled": float(self.y_scaled[idx]),
            "influence_weight": float(self.influence_weight[idx]),
            "estimation_weight": float(self.estimation_weight[idx]),
        }


def make_pair_collate_fn(tokenizer: Any, torch: Any):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        a_tokens = tokenizer.pad(
            [{"input_ids": item["input_ids_a"], "attention_mask": item["attention_mask_a"]} for item in batch],
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        b_tokens = tokenizer.pad(
            [{"input_ids": item["input_ids_b"], "attention_mask": item["attention_mask_b"]} for item in batch],
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        return {
            "input_ids_a": a_tokens["input_ids"],
            "attention_mask_a": a_tokens["attention_mask"],
            "input_ids_b": b_tokens["input_ids"],
            "attention_mask_b": b_tokens["attention_mask"],
            "features": torch.tensor(np.stack([item["features"] for item in batch]), dtype=torch.float32),
            "labels": torch.tensor([item["y_scaled"] for item in batch], dtype=torch.float32),
            "influence_weight": torch.tensor([item["influence_weight"] for item in batch], dtype=torch.float32),
            "estimation_weight": torch.tensor([item["estimation_weight"] for item in batch], dtype=torch.float32),
            "sample_id": torch.tensor([item["sample_id"] for item in batch], dtype=torch.long),
            "y_raw": torch.tensor([item["y_raw"] for item in batch], dtype=torch.float32),
        }

    return collate


def build_tokenizer(model_name: str, max_length: int, AutoTokenizer: Any) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_pair_model(
    config: dict[str, Any],
    n_features: int,
    torch: Any,
    AutoModel: Any,
    LoraConfig: Any,
    get_peft_model: Any,
    method: MethodSpec,
) -> Any:
    import torch.nn as nn

    class AntiSymmetricPairRegressionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            backbone = AutoModel.from_pretrained(
                config["model_name"],
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            if hasattr(backbone.config, "use_cache"):
                backbone.config.use_cache = False
            train_backbone = bool(method.train_backbone)
            self.train_backbone = train_backbone
            if train_backbone and bool(config.get("gradient_checkpointing", False)) and hasattr(backbone, "gradient_checkpointing_enable"):
                backbone.gradient_checkpointing_enable()
            if train_backbone and bool(config.get("gradient_checkpointing", False)) and hasattr(backbone, "enable_input_require_grads"):
                backbone.enable_input_require_grads()
            if train_backbone:
                lora_config = LoraConfig(
                    r=int(config["lora_r"]),
                    lora_alpha=int(config["lora_alpha"]),
                    target_modules=config["target_modules"],
                    lora_dropout=float(config["lora_dropout"]),
                    bias="none",
                )
                self.backbone = get_peft_model(backbone, lora_config)
            else:
                for param in backbone.parameters():
                    param.requires_grad_(False)
                self.backbone = backbone
            hidden_size = int(backbone.config.hidden_size)
            head_hidden_size = int(config.get("head_hidden_size", 128))
            if head_hidden_size > 0:
                self.head = nn.Sequential(
                    nn.Linear(hidden_size + int(n_features), head_hidden_size),
                    nn.ReLU(),
                    nn.Linear(head_hidden_size, 1),
                )
                last_layer = self.head[-1]
            else:
                self.head = nn.Linear(hidden_size + int(n_features), 1)
                last_layer = self.head
            nn.init.normal_(last_layer.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(last_layer.bias)
            self.pooling_strategy = str(config.get("pooling_strategy", "last")).lower()
            if self.pooling_strategy not in {"last", "mean"}:
                raise ValueError("pooling_strategy must be either 'last' or 'mean'")

        def pool(self, input_ids: Any, attention_mask: Any) -> Any:
            if self.train_backbone:
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            else:
                with torch.no_grad():
                    outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            if self.pooling_strategy == "last":
                lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
                return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths].float()
            mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return ((hidden * mask).sum(dim=1) / denom).float()

        def forward(
            self,
            input_ids_a: Any,
            attention_mask_a: Any,
            input_ids_b: Any,
            attention_mask_b: Any,
            features: Any,
        ) -> Any:
            h_a = self.pool(input_ids_a, attention_mask_a)
            h_b = self.pool(input_ids_b, attention_mask_b)
            z = torch.cat([h_a - h_b, features.float()], dim=1)
            return 0.5 * (self.head(z).squeeze(-1) - self.head(-z).squeeze(-1))

    return AntiSymmetricPairRegressionModel()


def _move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    skip = {"sample_id", "y_raw"}
    return {key: value.to(device) for key, value in batch.items() if key not in skip}


def capture_trainable_state(model: Any) -> dict[str, Any]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def restore_trainable_state(model: Any, state: dict[str, Any]) -> None:
    params = dict(model.named_parameters())
    for name, value in state.items():
        params[name].data.copy_(value.to(device=params[name].device, dtype=params[name].dtype))


def train_once(
    config: dict[str, Any],
    train_df: pd.DataFrame,
    stop_df: pd.DataFrame,
    scale_df: pd.DataFrame,
    y_scale: OutcomeScale,
    batch_size: int,
    seed: int,
    method: MethodSpec,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    mods = _require_training_modules()
    torch = mods["torch"]
    set_training_seeds(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LoRA training")
    device = torch.device("cuda")

    tokenizer = build_tokenizer(config["model_name"], int(config["max_length"]), mods["AutoTokenizer"])
    surrogate_feature_cols = surrogate_feature_cols_from_config(config)
    model = build_pair_model(
        config,
        len(surrogate_feature_cols),
        torch,
        mods["AutoModel"],
        mods["LoraConfig"],
        mods["get_peft_model"],
        method,
    ).to(device)
    try:
        model.backbone.print_trainable_parameters()
    except Exception:
        pass

    collate = make_pair_collate_fn(tokenizer, torch)
    train_dataset = PairTokenizedDataset(train_df, tokenizer, int(config["max_length"]), feature_cols=surrogate_feature_cols)
    train_loader = mods["DataLoader"](
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )
    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    lora_lr = float(method.lora_lr if method.lora_lr is not None else config["lora_lr"])
    head_lr = float(method.head_lr if method.head_lr is not None else config["head_lr"])
    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": float(config["weight_decay"])})
    param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": float(config["weight_decay"])})
    optimizer = torch.optim.AdamW(param_groups)
    max_epochs = int(method.max_epochs if method.max_epochs is not None else config.get("max_epochs", 12))
    min_epochs = int(method.min_epochs if method.min_epochs is not None else config.get("min_epochs", 3))
    patience = int(
        method.early_stopping_patience
        if method.early_stopping_patience is not None
        else config.get("early_stopping_patience", 3)
    )
    min_delta = float(config.get("early_stopping_min_delta", 0.0))
    best_state: dict[str, Any] | None = None
    best_value = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    total_steps = 0
    losses: list[float] = []
    epoch_history: list[dict[str, float | int | bool]] = []
    start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            inputs = _move_batch(batch, device)
            labels = inputs.pop("labels")
            influence_weight = inputs.pop("influence_weight")
            estimation_weight = inputs.pop("estimation_weight")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(**inputs)
            residual = labels.float() - pred.float()
            loss = training_loss_from_residuals(residual, influence_weight.float(), method, epoch, estimation_weight.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
            optimizer.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            epoch_losses.append(value)
            total_steps += 1

        stop_pred = predict_frame(model, tokenizer, stop_df, config, y_scale, torch, mods["DataLoader"], batch_size)
        stop_metrics = prediction_metrics(stop_pred)
        stop_value = stopping_value_from_metrics(stop_metrics, method.early_stopping_metric)
        improved = stop_value < best_value - min_delta
        if improved:
            best_value = stop_value
            best_epoch = epoch
            best_state = capture_trainable_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_history.append(
            {
                "epoch": int(epoch),
                "active_loss": active_loss_name(method, epoch),
                "train_loss_mean": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                "train_loss_last": float(epoch_losses[-1]) if epoch_losses else float("nan"),
                "validation_stop_mse": float(stop_metrics["rmse_scaled"]) ** 2,
                "validation_stop_ifvarq": float(stop_metrics["if_residual_var_scaled"]),
                "validation_stop_rmse": float(stop_metrics["rmse_scaled"]),
                "early_stopping_metric_value": float(stop_value),
                "is_best": bool(improved),
            }
        )
        if epoch % int(config.get("log_every_epochs", 1)) == 0 or improved:
            print(
                "train_progress",
                f"method={method.name}",
                f"loss={method.loss}",
                f"active_loss={epoch_history[-1]['active_loss']}",
                f"epoch={epoch}",
                f"train_loss={epoch_history[-1]['train_loss_mean']:.6f}",
                f"v_stop_mse={epoch_history[-1]['validation_stop_mse']:.6f}",
                f"v_stop_ifvarq={epoch_history[-1]['validation_stop_ifvarq']:.6f}",
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            early_stopped = True
            print("early_stop", f"method={method.name}", f"epoch={epoch}", f"best_epoch={best_epoch}", flush=True)
            break

    if best_state is not None:
        restore_trainable_state(model, best_state)
    stop_pred = predict_frame(model, tokenizer, stop_df, config, y_scale, torch, mods["DataLoader"], batch_size)
    scale_pred = predict_frame(model, tokenizer, scale_df, config, y_scale, torch, mods["DataLoader"], batch_size)
    runtime = {
        "runtime_seconds": time.time() - start,
        "actual_batch_size": int(batch_size),
        "requested_batch_size": int(method.batch_size if method.batch_size is not None else config["batch_size"]),
        "oom_fallback_used": bool(batch_size != int(method.batch_size if method.batch_size is not None else config["batch_size"])),
        "train_backbone": bool(method.train_backbone),
        "sampling_strategy": method.sampling_strategy,
        "warmup_loss": method.warmup_loss,
        "warmup_epochs": int(method.warmup_epochs),
        "if_weight_clip": method.if_weight_clip,
        "lora_lr": float(lora_lr),
        "head_lr": float(head_lr),
        "max_epochs": int(max_epochs),
        "min_epochs": int(min_epochs),
        "early_stopping_patience": int(patience),
        "early_stopping_min_delta": float(min_delta),
        "epochs_trained": int(epoch_history[-1]["epoch"]) if epoch_history else 0,
        "best_epoch": int(best_epoch),
        "early_stopped": bool(early_stopped),
        "steps_per_epoch": int(steps_per_epoch_for_s(len(train_df), batch_size)),
        "total_train_steps": int(total_steps),
        "best_validation_stop_metric_value": float(best_value),
        "final_train_loss": float(losses[-1]) if losses else float("nan"),
        "mean_train_loss": float(np.mean(losses)) if losses else float("nan"),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "epoch_history": epoch_history,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return runtime, stop_pred, scale_pred


def predict_frame(
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    config: dict[str, Any],
    y_scale: OutcomeScale,
    torch: Any,
    DataLoader: Any,
    batch_size: int,
) -> pd.DataFrame:
    model.eval()
    surrogate_feature_cols = surrogate_feature_cols_from_config(config)
    dataset = PairTokenizedDataset(frame, tokenizer, int(config["max_length"]), feature_cols=surrogate_feature_cols)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("prediction_batch_size", batch_size)),
        shuffle=False,
        collate_fn=make_pair_collate_fn(tokenizer, torch),
        num_workers=0,
        pin_memory=True,
    )
    rows: list[pd.DataFrame] = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            sample_ids = batch["sample_id"].cpu().numpy().astype(np.int64)
            y_raw_values = batch["y_raw"].cpu().numpy().astype(float)
            labels = batch["labels"].cpu().numpy().astype(float)
            influence_weight = batch["influence_weight"].cpu().numpy().astype(float)
            estimation_weight = batch["estimation_weight"].cpu().numpy().astype(float)
            inputs = _move_batch(batch, device)
            inputs.pop("labels")
            inputs.pop("influence_weight")
            inputs.pop("estimation_weight")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_scaled = model(**inputs).float().detach().cpu().numpy().astype(float)
            rows.append(
                pd.DataFrame(
                    {
                        ID_COL: sample_ids,
                        "y_raw": y_raw_values,
                        "y_scaled": labels,
                        "pred_scaled": pred_scaled,
                        "pred_raw": raw_y(pred_scaled, y_scale),
                        EST_WEIGHT_COL: estimation_weight,
                        IF_WEIGHT_COL: influence_weight,
                        LEGACY_IF_WEIGHT_COL: influence_weight,
                    }
                )
            )
    out = pd.concat(rows, ignore_index=True)
    out["residual_scaled"] = out["y_scaled"] - out["pred_scaled"]
    out["residual_raw"] = out["y_raw"] - out["pred_raw"]
    out["if_residual_scaled"] = out[IF_WEIGHT_COL] * out["residual_scaled"]
    out["if_residual_raw"] = out[IF_WEIGHT_COL] * out["residual_raw"]
    return out


def prediction_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    y = predictions["y_scaled"].to_numpy(dtype=float)
    pred = predictions["pred_scaled"].to_numpy(dtype=float)
    residual = y - pred
    if len(predictions) > 1 and np.std(y) > 0 and np.std(pred) > 0:
        corr = float(np.corrcoef(y, pred)[0, 1])
    else:
        corr = float("nan")
    return {
        "n": int(len(predictions)),
        "residual_mean_scaled": float(np.mean(residual)),
        "residual_var_scaled": float(np.var(residual, ddof=1)),
        "residual_var_raw": float(np.var(predictions["residual_raw"], ddof=1)),
        "rmse_scaled": float(np.sqrt(np.mean(residual**2))),
        "rmse_raw": float(np.sqrt(np.mean(np.asarray(predictions["residual_raw"], dtype=float) ** 2))),
        "prediction_var_scaled": float(np.var(pred, ddof=1)),
        "correlation": corr,
        "if_residual_mean_scaled": float(predictions["if_residual_scaled"].mean()),
        "if_residual_var_scaled": float(predictions["if_residual_scaled"].var(ddof=1)),
        "if_residual_mean_raw": float(predictions["if_residual_raw"].mean()),
        "if_residual_var_raw": float(predictions["if_residual_raw"].var(ddof=1)),
    }


def cell_dir(config: dict[str, Any], method_name: str, replication_id: int, s_train: int) -> Path:
    return Path(config["output_dir"]) / str(method_name).lower() / f"rep_{int(replication_id):02d}" / f"s_{int(s_train):04d}"


def write_cell_manifest(
    output_dir: Path,
    config: dict[str, Any],
    method: MethodSpec,
    replication_id: int,
    s_train: int,
    split: ScalingSplit,
    population: PopulationSplit,
) -> None:
    feature_cols = feature_cols_from_config(config)
    surrogate_feature_cols = surrogate_feature_cols_from_config(config)
    manifest = {
        "method": method.name,
        "loss": method.loss,
        "early_stopping_metric": method.early_stopping_metric,
        "sampling_strategy": method.sampling_strategy,
        "warmup_loss": method.warmup_loss,
        "warmup_epochs": int(method.warmup_epochs),
        "if_weight_clip": method.if_weight_clip,
        "train_backbone": bool(method.train_backbone),
        "lora_lr": method.lora_lr,
        "head_lr": method.head_lr,
        "batch_size": method.batch_size,
        "max_epochs": method.max_epochs,
        "min_epochs": method.min_epochs,
        "early_stopping_patience": method.early_stopping_patience,
        "replication_id": int(replication_id),
        "s_train": int(s_train),
        "target_feature": str(config.get("target_feature", TARGET_FEATURE)),
        "feature_columns": feature_cols,
        "surrogate_feature_columns": surrogate_feature_cols,
        "counts": {
            "H_scale": int(len(population.h_scale_ids)),
            "P_target": int(len(population.p_target_ids)),
            "train_pool": int(len(split.train_pool_ids)),
            "V_stop": int(len(split.v_stop_ids)),
            "V_scale": int(len(split.v_scale_ids)),
            "train": int(s_train),
        },
        "hashes": {
            "H_scale": ids_hash(population.h_scale_ids),
            "P_target": ids_hash(population.p_target_ids),
            "train_pool": ids_hash(split.train_pool_ids),
            "V_stop": ids_hash(split.v_stop_ids),
            "V_scale": ids_hash(split.v_scale_ids),
            "train": ids_hash(train_ids_for_s(split, s_train)),
        },
        "seed": rep_seed(config, replication_id),
    }
    write_json(output_dir / "split_manifest.json", manifest)


def cleanup_training_artifacts(output_dir: Path) -> None:
    for name in ("checkpoint", "checkpoints", "final_adapter"):
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)


def train_cell(config: dict[str, Any], method_name: str, replication_id: int, s_train: int) -> Path:
    method = method_by_name(config, method_name)
    output_dir = cell_dir(config, method.name, replication_id, s_train)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        print(f"skip_completed method={method.name} rep={replication_id} s={s_train} metrics={metrics_path}", flush=True)
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_upworthy_pairs(config["input_csv"], config)
    feature_cols = feature_cols_from_config(config)
    surrogate_feature_cols = surrogate_feature_cols_from_config(config)
    target_feature = str(config.get("target_feature", TARGET_FEATURE))
    population = build_population_split(df)
    y_scale = outcome_scale_from_h_scale(df, population.h_scale_ids)
    df, inference = compute_inference_setup(
        df,
        population.p_target_ids,
        y_scale,
        target_feature,
        feature_cols,
        hessian_ridge=float(config.get("hessian_ridge", 0.0) or 0.0),
    )
    split = build_scaling_split(df, config, replication_id, method.sampling_strategy, target_feature=target_feature)
    train_df = subset_by_ids(df, train_ids_for_s(split, s_train), "train", y_scale)
    stop_df = subset_by_ids(df, split.v_stop_ids, "validation_stop", y_scale)
    scale_df = subset_by_ids(df, split.v_scale_ids, "validation_scale", y_scale)
    constant_stop_metrics = prediction_metrics(constant_prediction_frame(stop_df, y_scale))
    constant_scale_metrics = prediction_metrics(constant_prediction_frame(scale_df, y_scale))
    write_cell_manifest(output_dir, config, method, replication_id, s_train, split, population)

    requested_batch = int(method.batch_size if method.batch_size is not None else config["batch_size"])
    batch_candidates = [requested_batch]
    for value in config.get("oom_fallback_batch_sizes", [64, 32]):
        batch = int(value)
        if batch > 0 and batch not in batch_candidates:
            batch_candidates.append(batch)

    for batch_index, batch_size in enumerate(batch_candidates):
        try:
            print(
                "upworthy_scaling_cell_start",
                f"method={method.name}",
                f"loss={method.loss}",
                f"sampling={method.sampling_strategy}",
                f"train_backbone={method.train_backbone}",
                f"rep={replication_id}",
                f"s={s_train}",
                f"batch_size={batch_size}",
                flush=True,
            )
            runtime, stop_pred, scale_pred = train_once(
                config,
                train_df,
                stop_df,
                scale_df,
                y_scale,
                batch_size=batch_size,
                seed=rep_seed(config, replication_id, salt=0 if bool(config.get("common_train_seed_within_rep", True)) else int(s_train)),
                method=method,
            )
            break
        except RuntimeError as exc:
            if not is_cuda_oom(exc) or batch_index == len(batch_candidates) - 1:
                raise
            next_batch = batch_candidates[batch_index + 1]
            print(f"cuda_oom_retry method={method.name} rep={replication_id} s={s_train} next_batch={next_batch}", flush=True)
            exc.__traceback__ = None
            del exc
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    stop_pred["replication_id"] = int(replication_id)
    stop_pred["s_train"] = int(s_train)
    stop_pred["split_role"] = "validation_stop"
    scale_pred["replication_id"] = int(replication_id)
    scale_pred["s_train"] = int(s_train)
    scale_pred["split_role"] = "validation_scale"
    if bool(config.get("save_predictions", False)):
        stop_pred.to_parquet(output_dir / "validation_stop_predictions.parquet", index=False)
        scale_pred.to_parquet(output_dir / "validation_scale_predictions.parquet", index=False)
    epoch_history = runtime.pop("epoch_history", [])
    if epoch_history:
        pd.DataFrame(epoch_history).to_csv(output_dir / "epoch_history.csv", index=False)

    metrics = {
        "method": method.name,
        "loss": method.loss,
        "early_stopping_metric": method.early_stopping_metric,
        "method_config": {
            "sampling_strategy": method.sampling_strategy,
            "warmup_loss": method.warmup_loss,
            "warmup_epochs": int(method.warmup_epochs),
            "if_weight_clip": method.if_weight_clip,
            "train_backbone": bool(method.train_backbone),
            "lora_lr": method.lora_lr,
            "head_lr": method.head_lr,
            "batch_size": method.batch_size,
            "max_epochs": method.max_epochs,
            "min_epochs": method.min_epochs,
            "early_stopping_patience": method.early_stopping_patience,
        },
        "replication_id": int(replication_id),
        "s_train": int(s_train),
        "model_name": config["model_name"],
        "input_csv": str(config["input_csv"]),
        "outcome_column": outcome_col_from_config(config),
        "outcome_transform": outcome_transform_from_config(config),
        "ctr_shrinkage_tau": config.get("ctr_shrinkage_tau"),
        "estimation_weight_scheme": estimation_weight_scheme_from_config(config),
        "pooling_strategy": str(config.get("pooling_strategy", "last")).lower(),
        "target_feature": target_feature,
        "feature_columns": feature_cols,
        "surrogate_feature_columns": surrogate_feature_cols,
        "h_scale_size": int(len(population.h_scale_ids)),
        "p_target_size": int(len(population.p_target_ids)),
        "train_pool_size": int(len(split.train_pool_ids)),
        "validation_stop_size": int(len(split.v_stop_ids)),
        "validation_scale_size": int(len(split.v_scale_ids)),
        "outcome_scale": {"mean": y_scale.mean, "sd": y_scale.sd},
        "inference": inference,
        "constant_validation_stop": constant_stop_metrics,
        "constant_validation_scale": constant_scale_metrics,
        "validation_stop": prediction_metrics(stop_pred),
        "validation_scale": prediction_metrics(scale_pred),
        "runtime": runtime,
    }
    write_json(metrics_path, metrics)
    cleanup_training_artifacts(output_dir)
    print(f"upworthy_scaling_cell_done method={method.name} rep={replication_id} s={s_train} metrics={metrics_path}", flush=True)
    return output_dir


def aggregate_task_cells(config: dict[str, Any]) -> list[tuple[str, int, int]]:
    cells: list[tuple[str, int, int]] = []
    for method in method_specs(config):
        for rep in replication_ids(config):
            for s_train in s_grid(config):
                cells.append((method.name, int(rep), int(s_train)))
    return cells


def task_index_to_cell(config: dict[str, Any], task_index: int) -> tuple[str, int, int]:
    cells = aggregate_task_cells(config)
    index = int(task_index)
    if index < 0 or index >= len(cells):
        raise ValueError(f"task_index={index} out of range for {len(cells)} cells")
    return cells[index]


def load_cell_metrics(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    missing = []
    for method_name, rep, s_train in aggregate_task_cells(config):
        path = cell_dir(config, method_name, rep, s_train) / "metrics.json"
        if not path.exists():
            missing.append(str(path))
            continue
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        method_config = metrics.get("method_config", {})
        runtime = metrics.get("runtime", {})
        feature_columns = metrics.get("feature_columns", FEATURE_COLS)
        surrogate_feature_columns = metrics.get("surrogate_feature_columns", feature_columns)
        constant_stop = metrics.get("constant_validation_stop", {})
        constant_scale = metrics.get("constant_validation_scale", {})
        row = {
            "method": str(metrics["method"]),
            "loss": str(metrics["loss"]),
            "early_stopping_metric": str(metrics["early_stopping_metric"]),
            "sampling_strategy": str(method_config.get("sampling_strategy", runtime.get("sampling_strategy", "uniform"))),
            "warmup_loss": method_config.get("warmup_loss", runtime.get("warmup_loss")),
            "warmup_epochs": int(method_config.get("warmup_epochs", runtime.get("warmup_epochs", 0)) or 0),
            "if_weight_clip": method_config.get("if_weight_clip", runtime.get("if_weight_clip")),
            "train_backbone": bool(method_config.get("train_backbone", runtime.get("train_backbone", True))),
            "lora_lr": runtime.get("lora_lr", method_config.get("lora_lr")),
            "head_lr": runtime.get("head_lr", method_config.get("head_lr")),
            "replication_id": int(metrics["replication_id"]),
            "s_train": int(metrics["s_train"]),
            "h_scale_size": int(metrics["h_scale_size"]),
            "p_target_size": int(metrics["p_target_size"]),
            "outcome_column": str(metrics.get("outcome_column", config.get("outcome_column", Y_COL))),
            "outcome_transform": str(metrics.get("outcome_transform", config.get("outcome_transform", ""))),
            "ctr_shrinkage_tau": metrics.get("ctr_shrinkage_tau", config.get("ctr_shrinkage_tau")),
            "estimation_weight_scheme": str(metrics.get("estimation_weight_scheme", config.get("estimation_weight_scheme", "none"))),
            "pooling_strategy": str(metrics.get("pooling_strategy", config.get("pooling_strategy", "last"))),
            "target_feature": str(metrics.get("target_feature", config.get("target_feature", TARGET_FEATURE))),
            "feature_columns": ",".join(str(x) for x in feature_columns),
            "n_features": int(len(feature_columns)),
            "surrogate_feature_columns": ",".join(str(x) for x in surrogate_feature_columns),
            "n_surrogate_features": int(len(surrogate_feature_columns)),
            "train_pool_size": int(metrics["train_pool_size"]),
            "validation_stop_size": int(metrics["validation_stop_size"]),
            "validation_scale_size": int(metrics["validation_scale_size"]),
            "question_beta_raw": float(metrics["inference"]["question_beta_raw"]),
            "target_coefficient_raw": float(metrics["inference"].get("target_coefficient_raw", metrics["inference"]["question_beta_raw"])),
            "direct_ols_ifvar_target_raw": float(metrics["inference"]["direct_ols_ifvar_target_raw"]),
            "direct_ols_ifvar_target_scaled": float(metrics["inference"]["direct_ols_ifvar_target_scaled"]),
            "hessian_condition_number": float(metrics["inference"].get("hessian_condition_number", float("nan"))),
            "hessian_ridge": float(metrics["inference"].get("hessian_ridge", float("nan"))),
            "if_weight_abs_p99": float(metrics["inference"].get("if_weight_abs_quantiles", {}).get("0.99", float("nan"))),
            "if_weight_abs_max": float(metrics["inference"].get("if_weight_abs_quantiles", {}).get("1.0", float("nan"))),
            "constant_validation_stop_ifvarq_raw": float(constant_stop.get("if_residual_var_raw", float("nan"))),
            "constant_validation_stop_ifvarq_scaled": float(constant_stop.get("if_residual_var_scaled", float("nan"))),
            "constant_validation_scale_ifvarq_raw": float(constant_scale.get("if_residual_var_raw", float("nan"))),
            "constant_validation_scale_ifvarq_scaled": float(constant_scale.get("if_residual_var_scaled", float("nan"))),
            "validation_stop_ifvarq_raw": float(metrics["validation_stop"]["if_residual_var_raw"]),
            "validation_stop_ifvarq_scaled": float(metrics["validation_stop"]["if_residual_var_scaled"]),
            "validation_stop_mse_scaled": float(metrics["validation_stop"]["rmse_scaled"]) ** 2,
            "validation_stop_rmse_scaled": float(metrics["validation_stop"]["rmse_scaled"]),
            "validation_stop_corr": float(metrics["validation_stop"]["correlation"]),
            "validation_scale_ifvarq_raw": float(metrics["validation_scale"]["if_residual_var_raw"]),
            "validation_scale_ifvarq_scaled": float(metrics["validation_scale"]["if_residual_var_scaled"]),
            "validation_scale_mse_scaled": float(metrics["validation_scale"]["rmse_scaled"]) ** 2,
            "validation_scale_rmse_scaled": float(metrics["validation_scale"]["rmse_scaled"]),
            "validation_scale_corr": float(metrics["validation_scale"]["correlation"]),
            "actual_batch_size": int(runtime["actual_batch_size"]),
            "oom_fallback_used": bool(runtime["oom_fallback_used"]),
            "runtime_seconds": float(runtime["runtime_seconds"]),
            "epochs_trained": int(runtime["epochs_trained"]),
            "best_epoch": int(runtime["best_epoch"]),
            "early_stopped": bool(runtime["early_stopped"]),
            "total_train_steps": int(runtime["total_train_steps"]),
            "device": str(runtime["device"]),
        }
        rows.append(row)
    if missing:
        raise FileNotFoundError("missing metrics.json files:\n" + "\n".join(missing[:20]))
    return pd.DataFrame(rows).sort_values(["method", "replication_id", "s_train"]).reset_index(drop=True)


def aggregate_scaling(config: dict[str, Any]) -> dict[str, Path]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_cell_metrics(config)
    metrics.to_csv(output_dir / "scaling_cell_metrics.csv", index=False)

    by_s = (
        metrics.groupby(
            [
                "outcome_column",
                "outcome_transform",
                "ctr_shrinkage_tau",
                "estimation_weight_scheme",
                "pooling_strategy",
                "target_feature",
                "feature_columns",
                "n_features",
                "surrogate_feature_columns",
                "n_surrogate_features",
                "method",
                "loss",
                "early_stopping_metric",
                "s_train",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_replications=("replication_id", "nunique"),
            mean_ifvarq_raw=("validation_scale_ifvarq_raw", "mean"),
            sd_ifvarq_raw=("validation_scale_ifvarq_raw", "std"),
            mean_ifvarq_scaled=("validation_scale_ifvarq_scaled", "mean"),
            mean_mse_scaled=("validation_scale_mse_scaled", "mean"),
            mean_rmse_scaled=("validation_scale_rmse_scaled", "mean"),
            mean_corr=("validation_scale_corr", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_epochs_trained=("epochs_trained", "mean"),
            direct_ols_ifvar_target_raw=("direct_ols_ifvar_target_raw", "first"),
            constant_ifvarq_raw=("constant_validation_scale_ifvarq_raw", "mean"),
            constant_ifvarq_scaled=("constant_validation_scale_ifvarq_scaled", "mean"),
            question_beta_raw=("question_beta_raw", "first"),
            target_coefficient_raw=("target_coefficient_raw", "first"),
            sampling_strategy=("sampling_strategy", "first"),
            train_backbone=("train_backbone", "first"),
            warmup_loss=("warmup_loss", "first"),
            warmup_epochs=("warmup_epochs", "first"),
            if_weight_clip=("if_weight_clip", "first"),
            lora_lr=("lora_lr", "first"),
            head_lr=("head_lr", "first"),
        )
        .sort_values(["method", "s_train"])
        .reset_index(drop=True)
    )
    by_s["se_ifvarq_raw"] = by_s["sd_ifvarq_raw"] / np.sqrt(by_s["n_replications"])
    by_s["ratio_to_constant_ifvarq"] = by_s["mean_ifvarq_raw"] / by_s["constant_ifvarq_raw"]
    by_s["drop_from_constant_pct"] = 100.0 * (1.0 - by_s["ratio_to_constant_ifvarq"])
    by_s["ratio_to_direct_ols_ifvarq"] = by_s["mean_ifvarq_raw"] / by_s["direct_ols_ifvar_target_raw"]
    by_s["drop_from_direct_ols_ifvarq_pct"] = 100.0 * (1.0 - by_s["ratio_to_direct_ols_ifvarq"])
    by_s.to_csv(output_dir / "scaling_by_s_summary.csv", index=False)

    fit_rows = []
    for method, group in by_s.groupby("method", sort=False):
        fit = fit_scaling_law(
            group["s_train"].to_numpy(dtype=float),
            group["mean_ifvarq_raw"].to_numpy(dtype=float),
            population_var_y=float(group["direct_ols_ifvar_target_raw"].iloc[0]),
        )
        fit_rows.append({"method": method, **fit})
    fits = pd.DataFrame(fit_rows)
    fits.to_csv(output_dir / "scaling_fit_parameters_raw.csv", index=False)

    break_even_rows = []
    budgets = [int(x) for x in config.get("diagnostic_budgets", [300, 500, 1000])]
    for _, row in by_s.iterrows():
        for budget in budgets:
            if int(row["s_train"]) >= budget:
                continue
            direct = float(row["direct_ols_ifvar_target_raw"]) / float(budget)
            ppi_proxy = float(row["mean_ifvarq_raw"]) / float(budget - int(row["s_train"]))
            break_even_rows.append(
                {
                    "method": row["method"],
                    "s_train": int(row["s_train"]),
                    "budget_B": int(budget),
                    "direct_labeled_only_var_proxy": direct,
                    "ppi_var_proxy": ppi_proxy,
                    "var_ratio_to_labeled_only": ppi_proxy / direct if direct > 0 else float("nan"),
                    "beats_labeled_only_proxy": bool(ppi_proxy < direct),
                }
            )
    break_even = pd.DataFrame(break_even_rows)
    break_even.to_csv(output_dir / "break_even_diagnostics.csv", index=False)
    return {
        "cell_metrics": output_dir / "scaling_cell_metrics.csv",
        "by_s_summary": output_dir / "scaling_by_s_summary.csv",
        "fit_parameters": output_dir / "scaling_fit_parameters_raw.csv",
        "break_even": output_dir / "break_even_diagnostics.csv",
    }


def describe_data(config: dict[str, Any]) -> dict[str, Any]:
    df = load_upworthy_pairs(config["input_csv"], config)
    feature_cols = feature_cols_from_config(config)
    surrogate_feature_cols = surrogate_feature_cols_from_config(config)
    target_feature = str(config.get("target_feature", TARGET_FEATURE))
    population = build_population_split(df)
    y_scale = outcome_scale_from_h_scale(df, population.h_scale_ids)
    df, inference = compute_inference_setup(
        df,
        population.p_target_ids,
        y_scale,
        target_feature,
        feature_cols,
        hessian_ridge=float(config.get("hessian_ridge", 0.0) or 0.0),
    )
    methods = method_specs(config)
    split0 = build_scaling_split(
        df,
        config,
        replication_ids(config)[0],
        methods[0].sampling_strategy,
        target_feature=target_feature,
    )
    return {
        "n_rows": int(len(df)),
        "h_scale_size": int(len(population.h_scale_ids)),
        "p_target_size": int(len(population.p_target_ids)),
        "train_pool_size": int(len(split0.train_pool_ids)),
        "validation_stop_size": int(len(split0.v_stop_ids)),
        "validation_scale_size": int(len(split0.v_scale_ids)),
        "s_grid": s_grid(config),
        "replication_ids": replication_ids(config),
        "methods": [method.__dict__ for method in methods],
        "target_feature": target_feature,
        "feature_columns": feature_cols,
        "n_features": int(len(feature_cols)),
        "surrogate_feature_columns": surrogate_feature_cols,
        "n_surrogate_features": int(len(surrogate_feature_cols)),
        "outcome_column": outcome_col_from_config(config),
        "outcome_transform": outcome_transform_from_config(config),
        "ctr_shrinkage_tau": config.get("ctr_shrinkage_tau"),
        "estimation_weight_scheme": estimation_weight_scheme_from_config(config),
        "pooling_strategy": str(config.get("pooling_strategy", "last")).lower(),
        "outcome_scale": {"mean": y_scale.mean, "sd": y_scale.sd},
        "question_beta_raw": inference["question_beta_raw"],
        "target_coefficient_raw": inference["target_coefficient_raw"],
        "hessian_condition_number": inference["hessian_condition_number"],
        "hessian_ridge": inference["hessian_ridge"],
        "if_weight_abs_quantiles": inference["if_weight_abs_quantiles"],
        "direct_ols_ifvar_target_raw": inference["direct_ols_ifvar_target_raw"],
        "direct_ols_ifvar_target_scaled": inference["direct_ols_ifvar_target_scaled"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upworthy question-coefficient scaling-law validation.")
    sub = parser.add_subparsers(dest="command", required=True)
    describe = sub.add_parser("describe")
    describe.add_argument("--config", required=True)
    train = sub.add_parser("train-cell")
    train.add_argument("--config", required=True)
    group = train.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-index", type=int)
    group.add_argument("--cell", nargs=3, metavar=("METHOD", "REP", "S"))
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "describe":
        print(json.dumps(describe_data(config), indent=2, sort_keys=True))
    elif args.command == "train-cell":
        if args.task_index is not None:
            method_name, rep, s_train = task_index_to_cell(config, args.task_index)
        else:
            method_name, rep, s_train = args.cell[0], int(args.cell[1]), int(args.cell[2])
        train_cell(config, method_name, int(rep), int(s_train))
    elif args.command == "aggregate":
        outputs = aggregate_scaling(config)
        print("upworthy_question_scaling_aggregate_done")
        for key, path in outputs.items():
            print(f"{key}: {path}")
    else:
        raise ValueError(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    main()
