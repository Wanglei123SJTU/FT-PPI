from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from src.data.upworthy_text_features import FEATURE_METADATA
from src.experiments.upworthy_question_scaling_law import (
    ID_COL,
    TEXT_A_COL,
    TEXT_B_COL,
    Y_COL,
    load_upworthy_pairs,
)


DEFAULT_CORE_CONTROLS = [
    "delta_QUESTION",
    "delta_NUMERIC",
    "delta_COMMON",
    "delta_SIMPLICITY",
    "delta_LENGTH",
    "delta_VADER_COMPOUND",
]
DEFAULT_S_GRID = [0, 50, 100, 250, 500, 750, 1000, 1500, 3000]
DEFAULT_BUDGETS = [500, 1000, 1500, 3000]
DEFAULT_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
TEXT_FEATURIZERS = {"word", "char", "word_char"}
SURROGATE_FEATURIZERS = {*TEXT_FEATURIZERS, "structured", "word_char_structured"}
TRAINING_OBJECTIVES = {"mse", "ifvar"}

LENGTH_READABILITY_FEATURES = {
    "delta_SIMPLICITY",
    "delta_LENGTH",
    "delta_READING_EASE",
    "delta_FK_GRADE",
    "delta_AVG_WORD_LENGTH",
    "delta_CHAR_LENGTH",
    "delta_LONG_WORD_SHARE",
}
HIGH_PRIORITY_FEATURES = {
    "QUESTION",
    "NUMERIC",
    "COMMON",
    "SIMPLICITY",
    "READING_EASE",
    "NEGATION",
    "VADER_POS",
    "VADER_NEG",
    "VADER_COMPOUND",
    "SENTIMENT_EXTREMITY",
    "SENTIMENT_INTENSITY",
}
BACKUP_ONLY_FEATURES = {
    "LENGTH",
    "CHAR_LENGTH",
    "QUESTION_MARKS",
    "EXCLAMATION",
    "EXCLAMATION_MARKS",
    "CAPS_SHARE",
    "HAS_QUOTES",
}


@dataclass(frozen=True)
class OlsStats:
    target: str
    feature_columns: list[str]
    beta: float
    abs_beta: float
    ifvar: float
    hessian_condition: float
    if_weight_p50: float
    if_weight_p90: float
    if_weight_p95: float
    if_weight_p99: float
    if_weight_max: float
    hessian_inv_target_row: np.ndarray


@dataclass(frozen=True)
class SurrogateFeaturizer:
    mode: str
    text_vectorizer: TfidfVectorizer | list[TfidfVectorizer] | None
    structured_columns: tuple[str, ...] = ()


def discover_candidate_features(df: pd.DataFrame, max_features: int | None = None) -> list[str]:
    candidates = [
        col
        for col in df.columns
        if col.startswith("delta_") and not col.endswith("_raw") and not col.endswith("_scale")
    ]
    candidates = [col for col in candidates if pd.api.types.is_numeric_dtype(df[col])]
    if max_features is not None:
        return candidates[: int(max_features)]
    return candidates


def core_controls_for_target(
    target: str,
    available_columns: Iterable[str],
    core_controls: Iterable[str] = DEFAULT_CORE_CONTROLS,
) -> list[str]:
    available = set(available_columns)
    controls = [col for col in core_controls if col in available and col != target]
    if target in LENGTH_READABILITY_FEATURES:
        controls = [col for col in controls if col not in {"delta_SIMPLICITY", "delta_LENGTH"}]
    return controls


def feature_set_for_target(target: str, available_columns: Iterable[str], core_controls: Iterable[str]) -> list[str]:
    return [target, *core_controls_for_target(target, available_columns, core_controls)]


def x_matrix(frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(len(frame), dtype=float), frame[feature_cols].to_numpy(dtype=float)])


def fit_ols_stats(frame: pd.DataFrame, target: str, feature_cols: list[str]) -> OlsStats:
    x = x_matrix(frame, feature_cols)
    y = frame[Y_COL].to_numpy(dtype=float)
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    hessian = (x.T @ x) / len(frame)
    singular_values = np.linalg.svd(hessian, compute_uv=False)
    min_singular = float(np.min(singular_values))
    max_singular = float(np.max(singular_values))
    condition = float(max_singular / min_singular) if min_singular > 0 else float("inf")
    hessian_inv = np.linalg.pinv(hessian)
    idx = 1 + feature_cols.index(target)
    if_weights = x @ hessian_inv[idx, :]
    residual = y - x @ beta
    if_residual = if_weights * residual
    quantiles = np.quantile(np.abs(if_weights), [0.5, 0.9, 0.95, 0.99, 1.0])
    return OlsStats(
        target=target,
        feature_columns=feature_cols,
        beta=float(beta[idx]),
        abs_beta=float(abs(beta[idx])),
        ifvar=float(np.var(if_residual, ddof=1)),
        hessian_condition=condition,
        if_weight_p50=float(quantiles[0]),
        if_weight_p90=float(quantiles[1]),
        if_weight_p95=float(quantiles[2]),
        if_weight_p99=float(quantiles[3]),
        if_weight_max=float(quantiles[4]),
        hessian_inv_target_row=hessian_inv[idx, :],
    )


def feature_name(delta_col: str) -> str:
    return delta_col.removeprefix("delta_")


def feature_priority(delta_col: str) -> int:
    name = feature_name(delta_col)
    if name in HIGH_PRIORITY_FEATURES or name.startswith("VADER_"):
        return 0
    if name in BACKUP_ONLY_FEATURES or delta_col in LENGTH_READABILITY_FEATURES:
        return 2
    return 1


def feature_source(delta_col: str) -> str:
    return FEATURE_METADATA.get(feature_name(delta_col), {}).get("source", "unknown")


def feature_note(delta_col: str) -> str:
    return FEATURE_METADATA.get(feature_name(delta_col), {}).get("note", "")


def pair_tfidf_matrix(vectorizer: TfidfVectorizer | list[TfidfVectorizer], frame: pd.DataFrame) -> sparse.csr_matrix:
    if isinstance(vectorizer, list):
        return sparse.hstack([pair_tfidf_matrix(item, frame) for item in vectorizer], format="csr")
    a = vectorizer.transform(frame[TEXT_A_COL].astype(str))
    b = vectorizer.transform(frame[TEXT_B_COL].astype(str))
    return (a - b).tocsr()


def fit_tfidf_vectorizer(frame: pd.DataFrame, max_features: int, min_df: int) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        strip_accents="unicode",
    )
    corpus = pd.concat([frame[TEXT_A_COL].astype(str), frame[TEXT_B_COL].astype(str)], ignore_index=True)
    vectorizer.fit(corpus)
    return vectorizer


def fit_text_featurizer(
    frame: pd.DataFrame,
    *,
    mode: str,
    max_features: int,
    min_df: int,
) -> TfidfVectorizer | list[TfidfVectorizer]:
    if mode not in TEXT_FEATURIZERS:
        raise ValueError(f"text featurizer must be one of {sorted(TEXT_FEATURIZERS)}, got {mode!r}")
    corpus = pd.concat([frame[TEXT_A_COL].astype(str), frame[TEXT_B_COL].astype(str)], ignore_index=True)
    if mode == "word":
        return fit_tfidf_vectorizer(frame, max_features=max_features, min_df=min_df)
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=min_df,
        max_features=max_features,
        strip_accents="unicode",
    )
    char.fit(corpus)
    if mode == "char":
        return char
    word = fit_tfidf_vectorizer(frame, max_features=max_features, min_df=min_df)
    return [word, char]


def fit_surrogate_featurizer(
    frame: pd.DataFrame,
    *,
    mode: str,
    max_features: int,
    min_df: int,
    structured_columns: Iterable[str] = (),
) -> SurrogateFeaturizer:
    if mode not in SURROGATE_FEATURIZERS:
        raise ValueError(f"surrogate featurizer must be one of {sorted(SURROGATE_FEATURIZERS)}, got {mode!r}")
    structured = tuple(str(col) for col in structured_columns)
    if mode == "structured":
        if not structured:
            raise ValueError("structured surrogate requires at least one structured column")
        return SurrogateFeaturizer(mode=mode, text_vectorizer=None, structured_columns=structured)
    if mode == "word_char_structured":
        if not structured:
            raise ValueError("word_char_structured surrogate requires at least one structured column")
        vectorizer = fit_text_featurizer(frame, mode="word_char", max_features=max_features, min_df=min_df)
        return SurrogateFeaturizer(mode=mode, text_vectorizer=vectorizer, structured_columns=structured)
    vectorizer = fit_text_featurizer(frame, mode=mode, max_features=max_features, min_df=min_df)
    return SurrogateFeaturizer(mode=mode, text_vectorizer=vectorizer)


def surrogate_matrix(featurizer: SurrogateFeaturizer, frame: pd.DataFrame) -> sparse.csr_matrix:
    parts: list[sparse.csr_matrix] = []
    if featurizer.text_vectorizer is not None:
        parts.append(pair_tfidf_matrix(featurizer.text_vectorizer, frame))
    if featurizer.structured_columns:
        missing = sorted(set(featurizer.structured_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"structured surrogate columns are missing: {missing}")
        structured = frame.loc[:, list(featurizer.structured_columns)].to_numpy(dtype=float)
        parts.append(sparse.csr_matrix(structured))
    if not parts:
        raise ValueError("surrogate featurizer produced no matrix parts")
    if len(parts) == 1:
        return parts[0].tocsr()
    return sparse.hstack(parts, format="csr")


def train_ridge_select_alpha(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    x_stop: sparse.csr_matrix,
    y_stop: np.ndarray,
    alphas: list[float],
) -> tuple[Ridge, float, float]:
    best_model: Ridge | None = None
    best_alpha = float(alphas[0])
    best_mse = float("inf")
    for alpha in alphas:
        model = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr")
        model.fit(x_train, y_train)
        mse = float(mean_squared_error(y_stop, model.predict(x_stop)))
        if mse < best_mse:
            best_model = model
            best_alpha = float(alpha)
            best_mse = mse
    if best_model is None:
        raise RuntimeError("no ridge model was fit")
    return best_model, best_alpha, best_mse


def _positive_sample_weights(values: np.ndarray, clip_quantile: float | None) -> np.ndarray:
    weights = np.square(np.asarray(values, dtype=float))
    if clip_quantile is not None and 0.0 < clip_quantile < 1.0 and len(weights) > 1:
        cap = float(np.quantile(weights, clip_quantile))
        if np.isfinite(cap) and cap > 0:
            weights = np.minimum(weights, cap)
    mean = float(np.mean(weights))
    if not np.isfinite(mean) or mean <= 0:
        return np.ones_like(weights, dtype=float)
    return weights / mean


def train_ridge_select_alpha_ifvar(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    if_weights_train: np.ndarray,
    x_stop: sparse.csr_matrix,
    y_stop: np.ndarray,
    if_weights_stop: np.ndarray,
    alphas: list[float],
    *,
    weight_clip_quantile: float | None = 0.99,
) -> tuple[Ridge, float, float]:
    best_model: Ridge | None = None
    best_alpha = float(alphas[0])
    best_metric = float("inf")
    sample_weight = _positive_sample_weights(if_weights_train, weight_clip_quantile)
    for alpha in alphas:
        model = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr")
        model.fit(x_train, y_train, sample_weight=sample_weight)
        residual = y_stop - model.predict(x_stop)
        metric = float(np.var(if_weights_stop * residual, ddof=1))
        if metric < best_metric:
            best_model = model
            best_alpha = float(alpha)
            best_metric = metric
    if best_model is None:
        raise RuntimeError("no target-aware ridge model was fit")
    return best_model, best_alpha, best_metric


def make_screening_split(
    n_rows: int,
    rng: np.random.Generator,
    train_pool_size: int,
    validation_stop_size: int,
    validation_scale_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    needed = train_pool_size + validation_stop_size + validation_scale_size
    if needed > n_rows:
        raise ValueError(f"screening split needs {needed} rows but H_scale only has {n_rows}")
    order = rng.permutation(n_rows)
    train = order[:train_pool_size]
    stop = order[train_pool_size : train_pool_size + validation_stop_size]
    scale = order[train_pool_size + validation_stop_size : needed]
    return train, stop, scale


def compute_candidate_summaries(
    df: pd.DataFrame,
    candidates: list[str],
    core_controls: list[str],
    budgets: list[int],
) -> tuple[pd.DataFrame, dict[str, OlsStats], dict[str, OlsStats]]:
    target = df.loc[df["split"].astype(str) == "target"].reset_index(drop=True)
    rows = []
    marginal_stats: dict[str, OlsStats] = {}
    controlled_stats: dict[str, OlsStats] = {}
    for candidate in candidates:
        marginal = fit_ols_stats(target, candidate, [candidate])
        controlled_cols = feature_set_for_target(candidate, candidates, core_controls)
        controlled = fit_ols_stats(target, candidate, controlled_cols)
        marginal_stats[candidate] = marginal
        controlled_stats[candidate] = controlled
        nonzero_share = float((df[candidate].to_numpy(dtype=float) != 0).mean())
        row = {
            "feature": feature_name(candidate),
            "delta_col": candidate,
            "source": feature_source(candidate),
            "note": feature_note(candidate),
            "nonzero_share": nonzero_share,
            "management_priority": feature_priority(candidate),
            "marginal_beta": marginal.beta,
            "marginal_abs_beta": marginal.abs_beta,
            "marginal_ifvar": marginal.ifvar,
            "controlled_beta": controlled.beta,
            "controlled_abs_beta": controlled.abs_beta,
            "controlled_ifvar": controlled.ifvar,
            "controlled_hessian_condition": controlled.hessian_condition,
            "if_weight_p50": controlled.if_weight_p50,
            "if_weight_p90": controlled.if_weight_p90,
            "if_weight_p95": controlled.if_weight_p95,
            "if_weight_p99": controlled.if_weight_p99,
            "if_weight_max": controlled.if_weight_max,
            "controlled_feature_columns": ",".join(controlled.feature_columns),
            "passes_stability": bool(
                nonzero_share >= 0.15
                and controlled.if_weight_p99 <= 10.0
                and math.isfinite(controlled.hessian_condition)
            ),
        }
        for budget in budgets:
            row[f"direct_ci_halfwidth_B{budget}"] = 1.96 * math.sqrt(max(controlled.ifvar, 0.0) / budget)
        rows.append(row)
    return pd.DataFrame(rows), marginal_stats, controlled_stats


def compute_tfidf_scaling(
    df: pd.DataFrame,
    candidates: list[str],
    controlled_stats: dict[str, OlsStats],
    *,
    s_values: list[int],
    replications: int,
    seed: int,
    train_pool_size: int,
    validation_stop_size: int,
    validation_scale_size: int,
    alphas: list[float],
    max_features: int,
    min_df: int,
    text_featurizer: str = "word",
    structured_columns: list[str] | None = None,
    training_objective: str = "mse",
    train_if_weight_clip_quantile: float | None = 0.99,
) -> pd.DataFrame:
    h_scale = df.loc[df["split"].astype(str) == "h_scale"].reset_index(drop=True)
    if h_scale.empty:
        raise ValueError("input must include split='h_scale'")
    if training_objective not in TRAINING_OBJECTIVES:
        raise ValueError(f"training_objective must be one of {sorted(TRAINING_OBJECTIVES)}, got {training_objective!r}")
    surrogate = fit_surrogate_featurizer(
        h_scale,
        mode=text_featurizer,
        max_features=max_features,
        min_df=min_df,
        structured_columns=structured_columns or [],
    )
    x_text = surrogate_matrix(surrogate, h_scale)
    y = h_scale[Y_COL].to_numpy(dtype=float)
    global_mean = float(np.mean(y))
    max_s = max(s_values)
    if train_pool_size < max_s:
        raise ValueError("train_pool_size must be at least max(s_grid)")

    weights_by_feature: dict[str, np.ndarray] = {}
    for candidate, stats in controlled_stats.items():
        x_cov = x_matrix(h_scale, stats.feature_columns)
        weights_by_feature[candidate] = x_cov @ stats.hessian_inv_target_row

    rows = []
    for rep in range(int(replications)):
        rng = np.random.default_rng(seed + rep)
        train_pool, stop_idx, scale_idx = make_screening_split(
            len(h_scale), rng, train_pool_size, validation_stop_size, validation_scale_size
        )
        y_scale = y[scale_idx]
        rep_ifvar0: dict[str, float] = {}
        for s in s_values:
            if s == 0:
                pred_scale = np.full(len(scale_idx), global_mean, dtype=float)
                selected_alpha = np.nan
                stop_mse = np.nan
                stop_metric = np.nan
                residual = y_scale - pred_scale
                for candidate in candidates:
                    if_residual = weights_by_feature[candidate][scale_idx] * residual
                    ifvar = float(np.var(if_residual, ddof=1))
                    rep_ifvar0[candidate] = ifvar
                    rows.append(
                        {
                            "replication": rep,
                            "s": int(s),
                            "feature": feature_name(candidate),
                            "delta_col": candidate,
                            "ifvar": ifvar,
                            "ifvar_ratio_to_s0": 1.0,
                            "selected_alpha": selected_alpha,
                            "stop_mse": stop_mse,
                            "stop_metric": stop_metric,
                            "text_featurizer": text_featurizer,
                            "training_objective": training_objective,
                        }
                    )
                continue
            if training_objective == "mse":
                train_idx = train_pool[:s]
                model, selected_alpha, stop_mse = train_ridge_select_alpha(
                    x_text[train_idx], y[train_idx], x_text[stop_idx], y[stop_idx], alphas
                )
                pred_scale = model.predict(x_text[scale_idx])
                residual = y_scale - pred_scale
                for candidate in candidates:
                    if_residual = weights_by_feature[candidate][scale_idx] * residual
                    ifvar = float(np.var(if_residual, ddof=1))
                    base = rep_ifvar0.get(candidate, np.nan)
                    rows.append(
                        {
                            "replication": rep,
                            "s": int(s),
                            "feature": feature_name(candidate),
                            "delta_col": candidate,
                            "ifvar": ifvar,
                            "ifvar_ratio_to_s0": float(ifvar / base) if np.isfinite(base) and base > 0 else np.nan,
                            "selected_alpha": selected_alpha,
                            "stop_mse": stop_mse,
                            "stop_metric": stop_mse,
                            "text_featurizer": text_featurizer,
                            "training_objective": training_objective,
                        }
                    )
            else:
                train_idx = train_pool[:s]
                for candidate in candidates:
                    if_weights = weights_by_feature[candidate]
                    model, selected_alpha, stop_metric = train_ridge_select_alpha_ifvar(
                        x_text[train_idx],
                        y[train_idx],
                        if_weights[train_idx],
                        x_text[stop_idx],
                        y[stop_idx],
                        if_weights[stop_idx],
                        alphas,
                        weight_clip_quantile=train_if_weight_clip_quantile,
                    )
                    pred_scale = model.predict(x_text[scale_idx])
                    stop_mse = float(mean_squared_error(y[stop_idx], model.predict(x_text[stop_idx])))
                    if_residual = if_weights[scale_idx] * (y_scale - pred_scale)
                    ifvar = float(np.var(if_residual, ddof=1))
                    base = rep_ifvar0.get(candidate, np.nan)
                    rows.append(
                        {
                            "replication": rep,
                            "s": int(s),
                            "feature": feature_name(candidate),
                            "delta_col": candidate,
                            "ifvar": ifvar,
                            "ifvar_ratio_to_s0": float(ifvar / base) if np.isfinite(base) and base > 0 else np.nan,
                            "selected_alpha": selected_alpha,
                            "stop_mse": stop_mse,
                            "stop_metric": stop_metric,
                            "text_featurizer": text_featurizer,
                            "training_objective": training_objective,
                        }
                    )
    return pd.DataFrame(rows)


def build_budget_win_table(scaling: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    grouped = (
        scaling.groupby(["delta_col", "feature", "s"], as_index=False)
        .agg(mean_ifvar=("ifvar", "mean"), mean_ratio=("ifvar_ratio_to_s0", "mean"), median_ratio=("ifvar_ratio_to_s0", "median"))
        .sort_values(["delta_col", "s"])
    )
    rows = []
    for (delta_col, feature), feature_rows in grouped.groupby(["delta_col", "feature"], sort=False):
        for budget in budgets:
            eligible = feature_rows.loc[(feature_rows["s"] > 0) & (feature_rows["s"] < budget)].copy()
            eligible["win_threshold"] = (budget - eligible["s"]) / budget
            winners = eligible.loc[eligible["mean_ratio"] < eligible["win_threshold"]].sort_values("s")
            if winners.empty:
                best = eligible.sort_values("mean_ratio").head(1)
                if best.empty:
                    rows.append(
                        {
                            "delta_col": delta_col,
                            "feature": feature,
                            "budget": budget,
                            "wins": False,
                            "best_s": np.nan,
                            "best_mean_ratio": np.nan,
                            "win_threshold_at_best_s": np.nan,
                        }
                    )
                else:
                    row = best.iloc[0]
                    rows.append(
                        {
                            "delta_col": delta_col,
                            "feature": feature,
                            "budget": budget,
                            "wins": False,
                            "best_s": int(row["s"]),
                            "best_mean_ratio": float(row["mean_ratio"]),
                            "win_threshold_at_best_s": float(row["win_threshold"]),
                        }
                    )
            else:
                row = winners.iloc[0]
                rows.append(
                    {
                        "delta_col": delta_col,
                        "feature": feature,
                        "budget": budget,
                        "wins": True,
                        "best_s": int(row["s"]),
                        "best_mean_ratio": float(row["mean_ratio"]),
                        "win_threshold_at_best_s": float(row["win_threshold"]),
                    }
                )
    return pd.DataFrame(rows)


def rank_candidates(summary: pd.DataFrame, budget_table: pd.DataFrame) -> pd.DataFrame:
    wins = budget_table.loc[budget_table["wins"]].sort_values(["delta_col", "budget", "best_s"])
    first_win = wins.groupby("delta_col", as_index=False).first()
    ranked = summary.merge(
        first_win[["delta_col", "budget", "best_s", "best_mean_ratio", "win_threshold_at_best_s"]].rename(
            columns={
                "budget": "smallest_winning_budget",
                "best_s": "best_s_at_smallest_winning_budget",
                "best_mean_ratio": "best_ratio_at_smallest_winning_budget",
                "win_threshold_at_best_s": "threshold_at_smallest_winning_budget",
            }
        ),
        on="delta_col",
        how="left",
    )
    ranked["smallest_winning_budget_sort"] = ranked["smallest_winning_budget"].fillna(np.inf)
    ranked = ranked.sort_values(
        [
            "passes_stability",
            "smallest_winning_budget_sort",
            "controlled_abs_beta",
            "management_priority",
            "if_weight_p99",
        ],
        ascending=[False, True, False, True, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked["recommended_top6"] = ranked["rank"] <= 6
    return ranked.drop(columns=["smallest_winning_budget_sort"])


def write_report(
    output_dir: Path,
    ranked: pd.DataFrame,
    budget_table: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    top = ranked.head(10)[
        [
            "rank",
            "feature",
            "source",
            "controlled_beta",
            "controlled_abs_beta",
            "nonzero_share",
            "if_weight_p99",
            "smallest_winning_budget",
            "best_s_at_smallest_winning_budget",
        ]
    ]
    lines = [
        "# Upworthy Feature Screening Report",
        "",
        f"- Input: `{args.input_csv}`",
        f"- Replications: {args.replications}",
        f"- s grid: {args.s_grid}",
        f"- budgets: {args.budgets}",
        f"- TF-IDF max features: {args.tfidf_max_features}",
        f"- surrogate featurizer: {args.text_featurizer}",
        f"- training objective: {args.training_objective}",
        "",
        "## Top Ranked Features",
        "",
        top.to_markdown(index=False),
        "",
        "## Budget Win Counts",
        "",
        budget_table.groupby("budget")["wins"].sum().reset_index(name="n_winning_features").to_markdown(index=False),
        "",
        "Interpretation: a budget win means at least one training size `s < B` has mean `nu_j(s)/nu_j(0) < (B-s)/B`.",
        "",
    ]
    (output_dir / "screening_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast CPU feature screening for Upworthy M-estimation targets.")
    parser.add_argument("--input-csv", type=Path, default=Path("Data/upworthy_pairs_with_text_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/upworthy_m_estimation/feature_screening"))
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--s-grid", type=int, nargs="+", default=DEFAULT_S_GRID)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--core-controls", nargs="+", default=DEFAULT_CORE_CONTROLS)
    parser.add_argument("--train-pool-size", type=int, default=3000)
    parser.add_argument("--validation-stop-size", type=int, default=1000)
    parser.add_argument("--validation-scale-size", type=int, default=1000)
    parser.add_argument("--tfidf-max-features", type=int, default=50000)
    parser.add_argument("--tfidf-min-df", type=int, default=3)
    parser.add_argument("--text-featurizer", choices=sorted(SURROGATE_FEATURIZERS), default="word")
    parser.add_argument("--training-objective", choices=sorted(TRAINING_OBJECTIVES), default="mse")
    parser.add_argument("--train-if-weight-clip-quantile", type=float, default=0.99)
    parser.add_argument("--candidate-features", nargs="+", default=None)
    parser.add_argument("--outcome-column", default=Y_COL)
    parser.add_argument("--outcome-transform", default="")
    parser.add_argument("--ctr-shrinkage-tau", type=float, default=0.0)
    parser.add_argument("--min-impressions-per-arm", type=float, default=None)
    parser.add_argument("--min-total-impressions", type=float, default=None)
    parser.add_argument("--min-clicks-per-arm", type=float, default=None)
    parser.add_argument("--max-features-to-screen", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_config = {
        "feature_columns": discover_candidate_features(pd.read_csv(args.input_csv, nrows=5)),
        "outcome_column": args.outcome_column,
        "outcome_transform": args.outcome_transform,
        "ctr_shrinkage_tau": args.ctr_shrinkage_tau,
        "min_impressions_per_arm": args.min_impressions_per_arm,
        "min_total_impressions": args.min_total_impressions,
        "min_clicks_per_arm": args.min_clicks_per_arm,
    }
    df = load_upworthy_pairs(args.input_csv, load_config)
    candidates = discover_candidate_features(df, max_features=args.max_features_to_screen)
    if args.candidate_features:
        requested = [
            str(item) if str(item).startswith("delta_") else f"delta_{item}"
            for item in args.candidate_features
        ]
        missing = sorted(set(requested) - set(candidates))
        if missing:
            raise ValueError(f"requested candidate features are unavailable: {missing}")
        candidates = requested
    if not candidates:
        raise ValueError("no candidate delta_* features found")
    summary, _, controlled_stats = compute_candidate_summaries(df, candidates, list(args.core_controls), list(args.budgets))
    scaling = compute_tfidf_scaling(
        df,
        candidates,
        controlled_stats,
        s_values=sorted(set(int(s) for s in args.s_grid)),
        replications=int(args.replications),
        seed=int(args.seed),
        train_pool_size=int(args.train_pool_size),
        validation_stop_size=int(args.validation_stop_size),
        validation_scale_size=int(args.validation_scale_size),
        alphas=[float(alpha) for alpha in args.alphas],
        max_features=int(args.tfidf_max_features),
        min_df=int(args.tfidf_min_df),
        text_featurizer=str(args.text_featurizer),
        structured_columns=candidates,
        training_objective=str(args.training_objective),
        train_if_weight_clip_quantile=float(args.train_if_weight_clip_quantile),
    )
    budget_table = build_budget_win_table(scaling, list(args.budgets))
    ranked = rank_candidates(summary, budget_table)

    summary.to_csv(args.output_dir / "candidate_screening_summary.csv", index=False)
    scaling.to_csv(args.output_dir / "tfidf_scaling_by_feature_s.csv", index=False)
    budget_table.to_csv(args.output_dir / "budget_win_table.csv", index=False)
    ranked.to_csv(args.output_dir / "ranked_shortlist.csv", index=False)
    (args.output_dir / "screening_args.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir, ranked, budget_table, args)
    print(json.dumps({"output_dir": str(args.output_dir), "n_features": len(candidates), "top_features": ranked.head(6)["feature"].tolist()}, indent=2))


if __name__ == "__main__":
    main()
