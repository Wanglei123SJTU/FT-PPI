from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


TEXT_COL = "description"
Y_RAW_COL = "points"
Y_COL = "points_scaled"
ID_COL = "sample_id"
SPLIT_COL = "split"

DEFAULT_COUNTRIES = [
    "US",
    "France",
    "Italy",
    "Spain",
    "Portugal",
    "Chile",
    "Argentina",
    "Austria",
    "Australia",
    "Germany",
]
DEFAULT_VARIETIES = [
    "Pinot Noir",
    "Chardonnay",
    "Cabernet Sauvignon",
    "Red Blend",
    "Bordeaux-style Red Blend",
    "Riesling",
    "Sauvignon Blanc",
    "Syrah",
    "Merlot",
    "Zinfandel",
]
DEFAULT_S_GRID = [0, 25, 50, 75, 100, 150, 200, 400, 800, 1500, 3000]
DEFAULT_BUDGETS = [500, 1000, 1500, 3000, 5000]
DEFAULT_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


@dataclass(frozen=True)
class FeatureInfo:
    column: str
    raw_column: str
    family: str
    note: str
    priority: int


@dataclass(frozen=True)
class OlsStats:
    target: str
    feature_columns: list[str]
    beta_scaled: float
    beta_raw_points: float
    ifvar: float
    residual_var: float
    hessian_condition: float
    if_weight_p50: float
    if_weight_p90: float
    if_weight_p95: float
    if_weight_p99: float
    if_weight_max: float
    hessian_inv_target_row: np.ndarray


@dataclass(frozen=True)
class IfvarRidgeModel:
    coef: np.ndarray
    centering_intercept: float
    alpha: float

    def predict(self, x: sparse.csr_matrix) -> np.ndarray:
        return np.asarray(self.coef[0] + x @ self.coef[1:]).reshape(-1)


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", str(text)))


def _sentence_count(text: str) -> int:
    count = len(re.findall(r"[.!?]+", str(text)))
    return max(1, count)


def _simple_syllables(word: str) -> int:
    token = re.sub(r"[^a-z]", "", word.lower())
    if not token:
        return 0
    groups = re.findall(r"[aeiouy]+", token)
    count = len(groups)
    if token.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _flesch_reading_ease(text: str) -> float:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", str(text))
    if not words:
        return 0.0
    syllables = sum(_simple_syllables(word) for word in words)
    sentences = _sentence_count(str(text))
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))


def _safe_vader_scores(texts: pd.Series) -> pd.DataFrame:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception:
        zeros = np.zeros(len(texts), dtype=float)
        return pd.DataFrame({"vader_compound": zeros, "vader_pos": zeros, "vader_neg": zeros})

    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(str(text)) for text in texts]
    return pd.DataFrame(
        {
            "vader_compound": [score["compound"] for score in scores],
            "vader_pos": [score["pos"] for score in scores],
            "vader_neg": [score["neg"] for score in scores],
        }
    )


def _standardize(series: pd.Series) -> tuple[pd.Series, float, float]:
    values = series.astype(float)
    mean = float(values.mean())
    sd = float(values.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.zeros(len(values), dtype=float), index=series.index), mean, sd
    return (values - mean) / sd, mean, sd


def _feature_infos(columns: Iterable[str]) -> dict[str, FeatureInfo]:
    infos: dict[str, FeatureInfo] = {}
    for column in columns:
        if column == "feat_log_price":
            infos[column] = FeatureInfo(
                column=column,
                raw_column="raw_log_price",
                family="price",
                note="Price-quality gradient: higher price associated with expert score.",
                priority=0,
            )
        elif column.startswith("feat_country_"):
            name = column.removeprefix("feat_country_")
            infos[column] = FeatureInfo(
                column=column,
                raw_column=f"raw_country_{name}",
                family="country",
                note=f"Origin premium/discount for wines from {name}.",
                priority=1,
            )
        elif column.startswith("feat_variety_"):
            name = column.removeprefix("feat_variety_")
            infos[column] = FeatureInfo(
                column=column,
                raw_column=f"raw_variety_{name}",
                family="variety",
                note=f"Varietal score premium/discount for {name}.",
                priority=1,
            )
        elif column == "feat_log_word_count":
            infos[column] = FeatureInfo(
                column=column,
                raw_column="raw_log_word_count",
                family="text_length",
                note="Review length/amount of description associated with expert score.",
                priority=2,
            )
        elif column == "feat_reading_ease":
            infos[column] = FeatureInfo(
                column=column,
                raw_column="raw_reading_ease",
                family="readability",
                note="Readability/simplicity of the review text associated with score.",
                priority=1,
            )
        elif column.startswith("feat_vader_"):
            name = column.removeprefix("feat_vader_")
            infos[column] = FeatureInfo(
                column=column,
                raw_column=f"raw_vader_{name}",
                family="sentiment",
                note=f"Open-source VADER {name} sentiment in the review text.",
                priority=1,
            )
        else:
            infos[column] = FeatureInfo(column=column, raw_column=column, family="other", note="", priority=3)
    return infos


def build_wine_screening_frame(
    input_csv: str | Path,
    *,
    seed: int = 20260620,
    experimental_population_size: int = 25000,
    h_scale_size: int = 10000,
    target_size: int = 15000,
    require_price: bool = True,
    countries: Iterable[str] = DEFAULT_COUNTRIES,
    varieties: Iterable[str] = DEFAULT_VARIETIES,
) -> tuple[pd.DataFrame, dict[str, FeatureInfo], dict]:
    df = pd.read_csv(input_csv)
    required = {TEXT_COL, Y_RAW_COL, "price", "country", "variety"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required wine columns: {missing}")

    clean = df[[TEXT_COL, Y_RAW_COL, "price", "country", "variety", "title"]].copy()
    clean = clean.dropna(subset=[TEXT_COL, Y_RAW_COL]).drop_duplicates(TEXT_COL, keep="first")
    if require_price:
        clean = clean.dropna(subset=["price"])
        clean = clean.loc[clean["price"].astype(float) > 0]
    clean = clean.reset_index(drop=True)
    if experimental_population_size > len(clean):
        raise ValueError(f"experimental_population_size={experimental_population_size} exceeds clean rows={len(clean)}")
    if h_scale_size + target_size > experimental_population_size:
        raise ValueError("h_scale_size + target_size must be <= experimental_population_size")

    out = clean.copy()
    out[TEXT_COL] = out[TEXT_COL].astype(str)
    out[Y_RAW_COL] = out[Y_RAW_COL].astype(float)
    out[Y_COL] = (out[Y_RAW_COL] - 90.0) / 5.0
    out["raw_log_price"] = np.log(out["price"].astype(float))
    out["raw_word_count"] = out[TEXT_COL].map(_word_count).astype(float)
    out["raw_log_word_count"] = np.log1p(out["raw_word_count"])
    out["raw_reading_ease"] = out[TEXT_COL].map(_flesch_reading_ease).astype(float)

    vader = _safe_vader_scores(out[TEXT_COL])
    out["raw_vader_compound"] = vader["vader_compound"].to_numpy(dtype=float)
    out["raw_vader_pos"] = vader["vader_pos"].to_numpy(dtype=float)
    out["raw_vader_neg"] = vader["vader_neg"].to_numpy(dtype=float)

    for country in countries:
        slug = _slug(country)
        out[f"raw_country_{slug}"] = (out["country"].astype(str) == country).astype(float)
    for variety in varieties:
        slug = _slug(variety)
        out[f"raw_variety_{slug}"] = (out["variety"].astype(str) == variety).astype(float)

    raw_to_feature: dict[str, str] = {
        "raw_log_price": "feat_log_price",
        "raw_log_word_count": "feat_log_word_count",
        "raw_reading_ease": "feat_reading_ease",
        "raw_vader_compound": "feat_vader_compound",
        "raw_vader_pos": "feat_vader_pos",
        "raw_vader_neg": "feat_vader_neg",
    }
    for country in countries:
        slug = _slug(country)
        raw_to_feature[f"raw_country_{slug}"] = f"feat_country_{slug}"
    for variety in varieties:
        slug = _slug(variety)
        raw_to_feature[f"raw_variety_{slug}"] = f"feat_variety_{slug}"

    scale_info = {}
    for raw_col, feature_col in raw_to_feature.items():
        out[feature_col], mean, sd = _standardize(out[raw_col])
        scale_info[feature_col] = {"raw_column": raw_col, "mean": mean, "sd": sd}

    rng = np.random.default_rng(seed)
    selected = rng.choice(out.index.to_numpy(), size=experimental_population_size, replace=False)
    out = out.loc[selected].reset_index(drop=True)
    out.insert(0, ID_COL, np.arange(len(out), dtype=np.int64))
    split = np.full(len(out), "unused", dtype=object)
    split[:h_scale_size] = "h_scale"
    split[h_scale_size : h_scale_size + target_size] = "target"
    out[SPLIT_COL] = split

    feature_cols = list(raw_to_feature.values())
    infos = _feature_infos(feature_cols)
    summary = {
        "input_rows": int(len(df)),
        "clean_rows_before_population_sample": int(len(clean)),
        "experimental_population_size": int(experimental_population_size),
        "h_scale_size": int(h_scale_size),
        "target_size": int(target_size),
        "require_price": bool(require_price),
        "feature_columns": feature_cols,
        "feature_scaling": "z-score on the cleaned wine population before population sampling",
        "y_scaling": "points_scaled = (points - 90) / 5",
        "scale_info": scale_info,
    }
    keep_cols = [
        ID_COL,
        SPLIT_COL,
        TEXT_COL,
        Y_RAW_COL,
        Y_COL,
        "price",
        "country",
        "variety",
        "title",
        *[info.raw_column for info in infos.values() if info.raw_column in out.columns],
        *feature_cols,
    ]
    return out.loc[:, keep_cols], infos, summary


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def x_matrix(frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(len(frame), dtype=float), frame[feature_cols].to_numpy(dtype=float)])


def fit_ols_stats(frame: pd.DataFrame, target: str, feature_cols: list[str], *, hessian_ridge: float = 0.0) -> OlsStats:
    if target not in feature_cols:
        raise ValueError("target must be included in feature_cols")
    x = x_matrix(frame, feature_cols)
    y = frame[Y_COL].to_numpy(dtype=float)
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    hessian = (x.T @ x) / len(frame)
    if hessian_ridge > 0:
        ridge = np.eye(hessian.shape[0], dtype=float)
        ridge[0, 0] = 0.0
        hessian_for_inverse = hessian + float(hessian_ridge) * ridge
    else:
        hessian_for_inverse = hessian
    singular = np.linalg.svd(hessian_for_inverse, compute_uv=False)
    min_s = float(np.min(singular))
    max_s = float(np.max(singular))
    condition = float(max_s / min_s) if min_s > 0 else float("inf")
    h_inv = np.linalg.pinv(hessian_for_inverse)
    target_idx = 1 + feature_cols.index(target)
    if_weight = x @ h_inv[target_idx, :]
    residual = y - x @ beta
    if_residual = if_weight * residual
    q = np.quantile(np.abs(if_weight), [0.5, 0.9, 0.95, 0.99, 1.0])
    return OlsStats(
        target=target,
        feature_columns=feature_cols,
        beta_scaled=float(beta[target_idx]),
        beta_raw_points=float(5.0 * beta[target_idx]),
        ifvar=float(np.var(if_residual, ddof=1)),
        residual_var=float(np.var(residual, ddof=1)),
        hessian_condition=condition,
        if_weight_p50=float(q[0]),
        if_weight_p90=float(q[1]),
        if_weight_p95=float(q[2]),
        if_weight_p99=float(q[3]),
        if_weight_max=float(q[4]),
        hessian_inv_target_row=h_inv[target_idx, :],
    )


def fit_text_vectorizer(frame: pd.DataFrame, max_features: int, min_df: int) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        strip_accents="unicode",
    )
    vectorizer.fit(frame[TEXT_COL].astype(str))
    return vectorizer


def train_ridge_mse(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    x_stop: sparse.csr_matrix,
    y_stop: np.ndarray,
    alphas: list[float],
    sample_weight: np.ndarray | None = None,
    if_weights_stop: np.ndarray | None = None,
) -> tuple[Ridge, float, float, float]:
    best_model: Ridge | None = None
    best_alpha = float(alphas[0])
    best_stop_mse = float("inf")
    best_stop_ifvar = float("inf")
    for alpha in alphas:
        model = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr")
        model.fit(x_train, y_train, sample_weight=sample_weight)
        pred_stop = model.predict(x_stop)
        stop_mse = float(mean_squared_error(y_stop, pred_stop))
        stop_ifvar = stop_mse
        if if_weights_stop is not None:
            stop_ifvar = float(np.var(if_weights_stop * (y_stop - pred_stop), ddof=1))
        score = stop_ifvar if if_weights_stop is not None else stop_mse
        best_score = best_stop_ifvar if if_weights_stop is not None else best_stop_mse
        if score < best_score:
            best_model = model
            best_alpha = float(alpha)
            best_stop_mse = stop_mse
            best_stop_ifvar = stop_ifvar
    if best_model is None:
        raise RuntimeError("no ridge model was selected")
    return best_model, best_alpha, best_stop_mse, best_stop_ifvar


def train_ridge_ifvar_exact(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    if_weights_train: np.ndarray,
    x_stop: sparse.csr_matrix,
    y_stop: np.ndarray,
    if_weights_stop: np.ndarray,
    alphas: list[float],
) -> tuple[IfvarRidgeModel, float, float, float]:
    """Fit a linear surrogate for the exact sample IF-variance objective.

    The objective is

        min_f Var_n[w_i {Y_i - f(T_i)}].

    This is equivalent to fitting a ridge model to the transformed regression
    target w_i Y_i and transformed design w_i [1, T_i], with an additional
    free intercept absorbing the sample mean of w_i residuals. The returned
    model predicts on the original scale as f(T)=theta_0 + T theta.
    """
    w_train = np.asarray(if_weights_train, dtype=float)
    w_stop = np.asarray(if_weights_stop, dtype=float)
    x_aug_train = sparse.hstack(
        [sparse.csr_matrix(w_train.reshape(-1, 1)), x_train.multiply(w_train[:, None])],
        format="csr",
    )
    y_aug_train = w_train * np.asarray(y_train, dtype=float)

    best: IfvarRidgeModel | None = None
    best_alpha = float(alphas[0])
    best_stop_mse = float("inf")
    best_stop_ifvar = float("inf")
    for alpha in alphas:
        ridge = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr")
        ridge.fit(x_aug_train, y_aug_train)
        model = IfvarRidgeModel(
            coef=np.asarray(ridge.coef_, dtype=float).reshape(-1),
            centering_intercept=float(ridge.intercept_),
            alpha=float(alpha),
        )
        pred_stop = model.predict(x_stop)
        stop_mse = float(mean_squared_error(y_stop, pred_stop))
        stop_ifvar = float(np.var(w_stop * (y_stop - pred_stop), ddof=1))
        if stop_ifvar < best_stop_ifvar:
            best = model
            best_alpha = float(alpha)
            best_stop_mse = stop_mse
            best_stop_ifvar = stop_ifvar
    if best is None:
        raise RuntimeError("no exact IFVar ridge model was selected")
    return best, best_alpha, best_stop_mse, best_stop_ifvar


def positive_ifvar_weights(if_weights: np.ndarray, clip_quantile: float | None = 0.99) -> np.ndarray:
    weights = np.square(np.asarray(if_weights, dtype=float))
    if clip_quantile is not None and 0.0 < clip_quantile < 1.0 and len(weights) > 1:
        cap = float(np.quantile(weights, clip_quantile))
        if np.isfinite(cap) and cap > 0:
            weights = np.minimum(weights, cap)
    mean = float(np.mean(weights))
    if not np.isfinite(mean) or mean <= 0:
        return np.ones_like(weights, dtype=float)
    return weights / mean


def make_nested_split(
    n_rows: int,
    rng: np.random.Generator,
    train_pool_size: int,
    validation_stop_size: int,
    validation_scale_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    needed = train_pool_size + validation_stop_size + validation_scale_size
    if needed > n_rows:
        raise ValueError(f"need {needed} h_scale rows, found {n_rows}")
    order = rng.permutation(n_rows)
    return (
        order[:train_pool_size],
        order[train_pool_size : train_pool_size + validation_stop_size],
        order[train_pool_size + validation_stop_size : needed],
    )


def discover_feature_columns(df: pd.DataFrame, max_features: int | None = None) -> list[str]:
    cols = [col for col in df.columns if col.startswith("feat_") and pd.api.types.is_numeric_dtype(df[col])]
    usable = [col for col in cols if float(df[col].std(ddof=0)) > 1e-12]
    if max_features is not None:
        usable = usable[: int(max_features)]
    return usable


def compute_candidate_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    infos: dict[str, FeatureInfo],
    budgets: list[int],
    *,
    controls: list[str] | None = None,
    hessian_ridge: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, OlsStats]]:
    target = df.loc[df[SPLIT_COL] == "target"].reset_index(drop=True)
    controls = [col for col in (controls or []) if col in feature_cols]
    rows = []
    stats_by_feature: dict[str, OlsStats] = {}
    for feature in feature_cols:
        active = [feature, *[col for col in controls if col != feature]]
        stats = fit_ols_stats(target, feature, active, hessian_ridge=hessian_ridge)
        stats_by_feature[feature] = stats
        raw_col = infos.get(feature, FeatureInfo(feature, feature, "other", "", 3)).raw_column
        raw = df[raw_col].to_numpy(dtype=float) if raw_col in df.columns else df[feature].to_numpy(dtype=float)
        raw_sd = float(np.std(raw, ddof=0))
        beta_raw_unit_points = float(stats.beta_raw_points / raw_sd) if np.isfinite(raw_sd) and raw_sd > 0 else float("nan")
        nonzero_share = float((np.abs(raw) > 1e-12).mean())
        row = {
            "feature": feature,
            "raw_feature": raw_col,
            "family": infos.get(feature, FeatureInfo(feature, feature, "other", "", 3)).family,
            "note": infos.get(feature, FeatureInfo(feature, feature, "other", "", 3)).note,
            "priority": infos.get(feature, FeatureInfo(feature, feature, "other", "", 3)).priority,
            "nonzero_share": nonzero_share,
            "raw_mean": float(np.mean(raw)),
            "raw_sd": raw_sd,
            "raw_p05": float(np.quantile(raw, 0.05)),
            "raw_p50": float(np.quantile(raw, 0.50)),
            "raw_p95": float(np.quantile(raw, 0.95)),
            "beta_scaled_y": stats.beta_scaled,
            "beta_raw_points": stats.beta_raw_points,
            "beta_raw_unit_points": beta_raw_unit_points,
            "abs_beta_raw_points": abs(stats.beta_raw_points),
            "abs_beta_raw_unit_points": abs(beta_raw_unit_points),
            "direct_ifvar": stats.ifvar,
            "direct_residual_var": stats.residual_var,
            "hessian_condition": stats.hessian_condition,
            "if_weight_p50": stats.if_weight_p50,
            "if_weight_p90": stats.if_weight_p90,
            "if_weight_p95": stats.if_weight_p95,
            "if_weight_p99": stats.if_weight_p99,
            "if_weight_max": stats.if_weight_max,
            "active_regression_features": ",".join(active),
            "passes_stability": bool(nonzero_share >= 0.03 and stats.if_weight_p99 <= 10.0 and math.isfinite(stats.hessian_condition)),
        }
        for budget in budgets:
            row[f"direct_ci_halfwidth_points_B{budget}"] = 1.96 * 5.0 * math.sqrt(max(stats.ifvar, 0.0) / budget)
        rows.append(row)
    return pd.DataFrame(rows), stats_by_feature


def compute_tfidf_scaling(
    df: pd.DataFrame,
    feature_cols: list[str],
    stats_by_feature: dict[str, OlsStats],
    *,
    s_grid: list[int],
    replications: int,
    seed: int,
    train_pool_size: int,
    validation_stop_size: int,
    validation_scale_size: int,
    alphas: list[float],
    max_tfidf_features: int,
    min_df: int,
    if_weight_clip_quantile: float | None,
) -> pd.DataFrame:
    h_scale = df.loc[df[SPLIT_COL] == "h_scale"].reset_index(drop=True)
    vectorizer = fit_text_vectorizer(h_scale, max_features=max_tfidf_features, min_df=min_df)
    x_text = vectorizer.transform(h_scale[TEXT_COL].astype(str)).tocsr()
    y = h_scale[Y_COL].to_numpy(dtype=float)
    global_mean = float(np.mean(y))
    weights_by_feature: dict[str, np.ndarray] = {}
    for feature, stats in stats_by_feature.items():
        weights_by_feature[feature] = x_matrix(h_scale, stats.feature_columns) @ stats.hessian_inv_target_row

    rows = []
    max_s = max(s_grid)
    if train_pool_size < max_s:
        raise ValueError("train_pool_size must be >= max(s_grid)")
    for rep in range(int(replications)):
        rng = np.random.default_rng(seed + 7919 * rep)
        train_pool, stop_idx, scale_idx = make_nested_split(
            len(h_scale),
            rng,
            train_pool_size=train_pool_size,
            validation_stop_size=validation_stop_size,
            validation_scale_size=validation_scale_size,
        )
        y_scale = y[scale_idx]
        s0_by_feature: dict[str, float] = {}
        for s in s_grid:
            if s == 0:
                residual = y_scale - global_mean
                for feature in feature_cols:
                    if_resid = weights_by_feature[feature][scale_idx] * residual
                    ifvar = float(np.var(if_resid, ddof=1))
                    s0_by_feature[feature] = ifvar
                    rows.append(_scaling_row(rep, s, feature, "constant", ifvar, ifvar, np.nan, np.nan, np.nan))
                continue
            train_idx = train_pool[:s]
            uniform_model, uniform_alpha, uniform_stop_mse, uniform_stop_ifvar = train_ridge_mse(
                x_text[train_idx],
                y[train_idx],
                x_text[stop_idx],
                y[stop_idx],
                alphas,
            )
            pred_uniform = uniform_model.predict(x_text[scale_idx])
            uniform_resid = y_scale - pred_uniform
            for feature in feature_cols:
                if_weight = weights_by_feature[feature]
                ifvar = float(np.var(if_weight[scale_idx] * uniform_resid, ddof=1))
                rows.append(
                    _scaling_row(
                        rep,
                        s,
                        feature,
                        "mse",
                        ifvar,
                        s0_by_feature.get(feature, np.nan),
                        uniform_alpha,
                        uniform_stop_mse,
                        uniform_stop_ifvar,
                    )
                )
                mse_ifvar_model, mse_ifvar_alpha, mse_ifvar_stop_mse, mse_ifvar_stop_ifvar = train_ridge_mse(
                    x_text[train_idx],
                    y[train_idx],
                    x_text[stop_idx],
                    y[stop_idx],
                    alphas,
                    if_weights_stop=if_weight[stop_idx],
                )
                mse_ifvar_pred = mse_ifvar_model.predict(x_text[scale_idx])
                mse_ifvar = float(np.var(if_weight[scale_idx] * (y_scale - mse_ifvar_pred), ddof=1))
                rows.append(
                    _scaling_row(
                        rep,
                        s,
                        feature,
                        "mse_stop_ifvar",
                        mse_ifvar,
                        s0_by_feature.get(feature, np.nan),
                        mse_ifvar_alpha,
                        mse_ifvar_stop_mse,
                        mse_ifvar_stop_ifvar,
                    )
                )
            for feature in feature_cols:
                if_weight = weights_by_feature[feature]
                sample_weight = positive_ifvar_weights(if_weight[train_idx], clip_quantile=if_weight_clip_quantile)
                model, alpha, stop_mse, stop_ifvar = train_ridge_mse(
                    x_text[train_idx],
                    y[train_idx],
                    x_text[stop_idx],
                    y[stop_idx],
                    alphas,
                    sample_weight=sample_weight,
                    if_weights_stop=if_weight[stop_idx],
                )
                pred = model.predict(x_text[scale_idx])
                ifvar = float(np.var(if_weight[scale_idx] * (y_scale - pred), ddof=1))
                rows.append(
                    _scaling_row(
                        rep,
                        s,
                        feature,
                        "ifvar_weighted",
                        ifvar,
                        s0_by_feature.get(feature, np.nan),
                        alpha,
                        stop_mse,
                        stop_ifvar,
                    )
                )
                exact_model, exact_alpha, exact_stop_mse, exact_stop_ifvar = train_ridge_ifvar_exact(
                    x_text[train_idx],
                    y[train_idx],
                    if_weight[train_idx],
                    x_text[stop_idx],
                    y[stop_idx],
                    if_weight[stop_idx],
                    alphas,
                )
                exact_pred = exact_model.predict(x_text[scale_idx])
                exact_ifvar = float(np.var(if_weight[scale_idx] * (y_scale - exact_pred), ddof=1))
                rows.append(
                    _scaling_row(
                        rep,
                        s,
                        feature,
                        "ifvar_exact",
                        exact_ifvar,
                        s0_by_feature.get(feature, np.nan),
                        exact_alpha,
                        exact_stop_mse,
                        exact_stop_ifvar,
                    )
                )
    return pd.DataFrame(rows)


def _scaling_row(
    rep: int,
    s: int,
    feature: str,
    objective: str,
    ifvar: float,
    ifvar0: float,
    alpha: float,
    stop_mse: float,
    stop_ifvar: float,
) -> dict:
    return {
        "replication": int(rep),
        "s": int(s),
        "feature": feature,
        "training_objective": objective,
        "ifvar": float(ifvar),
        "ifvar_ratio_to_s0": float(ifvar / ifvar0) if np.isfinite(ifvar0) and ifvar0 > 0 else np.nan,
        "selected_alpha": alpha,
        "stop_mse": stop_mse,
        "stop_ifvar": stop_ifvar,
    }


def build_budget_win_table(scaling: pd.DataFrame, candidate_summary: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    direct = candidate_summary.set_index("feature")["direct_ifvar"].to_dict()
    rows = []
    for (feature, objective), group in scaling.groupby(["feature", "training_objective"]):
        if objective == "constant":
            continue
        direct_ifvar = float(direct[feature])
        mean_curve = group.groupby("s", as_index=False)["ifvar"].mean()
        for budget in budgets:
            feasible = mean_curve.loc[mean_curve["s"] < int(budget)].copy()
            feasible["variance_ratio_vs_direct"] = (feasible["ifvar"] / (int(budget) - feasible["s"])) / (
                direct_ifvar / int(budget)
            )
            best = feasible.loc[feasible["variance_ratio_vs_direct"].idxmin()]
            wins = bool(best["variance_ratio_vs_direct"] < 1.0)
            rows.append(
                {
                    "feature": feature,
                    "training_objective": objective,
                    "budget": int(budget),
                    "best_s": int(best["s"]),
                    "best_ifvar": float(best["ifvar"]),
                    "variance_ratio_vs_direct": float(best["variance_ratio_vs_direct"]),
                    "wins": wins,
                    "win_margin_pct": float((1.0 - best["variance_ratio_vs_direct"]) * 100.0),
                }
            )
    return pd.DataFrame(rows).sort_values(["budget", "variance_ratio_vs_direct", "feature"]).reset_index(drop=True)


def fit_power_law_by_feature(scaling: pd.DataFrame) -> pd.DataFrame:
    from src.experiments.wine_var_scaling_law import fit_scaling_law

    rows = []
    for (feature, objective), group in scaling.loc[scaling["s"] > 0].groupby(["feature", "training_objective"]):
        curve = group.groupby("s", as_index=False)["ifvar"].mean().sort_values("s")
        if len(curve) < 4:
            continue
        try:
            fit = fit_scaling_law(curve["s"].to_numpy(dtype=float), curve["ifvar"].to_numpy(dtype=float), population_var_y=curve["ifvar"].max())
        except Exception:
            continue
        rows.append({"feature": feature, "training_objective": objective, **fit})
    return pd.DataFrame(rows)


def build_ranked_shortlist(candidate_summary: pd.DataFrame, budget: pd.DataFrame) -> pd.DataFrame:
    best = (
        budget.groupby(["feature", "training_objective"], as_index=False)
        .agg(best_ratio=("variance_ratio_vs_direct", "min"), wins_any=("wins", "max"))
        .sort_values(["feature", "best_ratio"])
    )
    pivot = best.pivot(index="feature", columns="training_objective", values="best_ratio").reset_index()
    wins = best.groupby("feature", as_index=False)["wins_any"].max().rename(columns={"wins_any": "wins_any_budget"})
    out = candidate_summary.merge(pivot, on="feature", how="left").merge(wins, on="feature", how="left")
    mse_baseline_cols = [col for col in ["mse", "mse_stop_ifvar"] if col in out.columns]
    mse_baseline = out[mse_baseline_cols].min(axis=1) if mse_baseline_cols else np.inf
    if "ifvar_exact" in out.columns:
        out["ifvar_beats_mse"] = out["ifvar_exact"] < mse_baseline
    else:
        out["ifvar_beats_mse"] = out.get("ifvar_weighted", np.inf) < mse_baseline
    out["rank_key"] = (
        (~out["passes_stability"]).astype(int) * 100
        + (~out["wins_any_budget"].fillna(False).astype(bool)).astype(int) * 20
        + out["priority"].astype(int)
        - out["abs_beta_raw_points"].rank(method="dense", ascending=False) / 100.0
    )
    return out.sort_values(["rank_key", "ifvar_weighted", "mse", "feature"]).reset_index(drop=True)


def write_report(
    output_dir: Path,
    data_summary: dict,
    ranked: pd.DataFrame,
    budget: pd.DataFrame,
    scaling_fit: pd.DataFrame,
) -> None:
    lines = [
        "# Wine Coefficient Screening Report",
        "",
        "This is a CPU feasibility screen for replacing mean estimation with a single OLS coefficient target.",
        "The surrogate sees only wine review descriptions. The target regression is low-dimensional OLS on wine metadata or text-derived features.",
        "",
        "## Data",
        "",
        f"- Experimental population: `{data_summary['experimental_population_size']}`",
        f"- H-scale rows: `{data_summary['h_scale_size']}`",
        f"- Target rows: `{data_summary['target_size']}`",
        f"- Y scaling: `{data_summary['y_scaling']}`",
        "",
        "## Top Shortlist",
        "",
    ]
    show_cols = [
        "feature",
        "family",
        "beta_raw_points",
        "beta_raw_unit_points",
        "direct_ifvar",
        "if_weight_p99",
        "mse",
        "mse_stop_ifvar",
        "ifvar_exact",
        "ifvar_weighted",
        "ifvar_beats_mse",
        "note",
    ]
    available = [col for col in show_cols if col in ranked.columns]
    lines.append(ranked.loc[:, available].head(12).to_markdown(index=False))
    lines.extend(["", "## Best Budget Cells", ""])
    lines.append(
        budget.sort_values("variance_ratio_vs_direct")
        .head(20)
        .to_markdown(index=False)
    )
    lines.extend(["", "## Scaling Fit", ""])
    if len(scaling_fit):
        fit_cols = [col for col in ["feature", "training_objective", "a", "alpha", "b", "r2"] if col in scaling_fit.columns]
        lines.append(scaling_fit.loc[:, fit_cols].sort_values("r2", ascending=False).head(20).to_markdown(index=False))
    else:
        lines.append("No power-law fit was available.")
    (output_dir / "screening_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast wine review single-coefficient screening.")
    parser.add_argument("--input-csv", default="Data/wine_data.csv")
    parser.add_argument("--output-dir", default="artifacts/wine_coefficient_screening")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--experimental-population-size", type=int, default=25000)
    parser.add_argument("--h-scale-size", type=int, default=10000)
    parser.add_argument("--target-size", type=int, default=15000)
    parser.add_argument("--max-features-to-screen", type=int)
    parser.add_argument("--feature-cols", default="")
    parser.add_argument("--controls", default="")
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--s-grid", default=",".join(map(str, DEFAULT_S_GRID)))
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--alphas", default=",".join(map(str, DEFAULT_ALPHAS)))
    parser.add_argument("--train-pool-size", type=int, default=5000)
    parser.add_argument("--validation-stop-size", type=int, default=1000)
    parser.add_argument("--validation-scale-size", type=int, default=2000)
    parser.add_argument("--max-tfidf-features", type=int, default=50000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--if-weight-clip-quantile", type=float, default=0.99)
    parser.add_argument("--hessian-ridge", type=float, default=0.0)
    return parser.parse_args()


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, infos, data_summary = build_wine_screening_frame(
        args.input_csv,
        seed=args.seed,
        experimental_population_size=args.experimental_population_size,
        h_scale_size=args.h_scale_size,
        target_size=args.target_size,
    )
    feature_cols = _parse_csv_strings(args.feature_cols) or discover_feature_columns(df, args.max_features_to_screen)
    unknown = sorted(set(feature_cols) - set(df.columns))
    if unknown:
        raise ValueError(f"unknown feature columns: {unknown}")
    controls = _parse_csv_strings(args.controls)
    budgets = _parse_csv_ints(args.budgets)
    s_grid = _parse_csv_ints(args.s_grid)
    alphas = _parse_csv_floats(args.alphas)

    summary, stats = compute_candidate_summary(
        df,
        feature_cols,
        infos,
        budgets,
        controls=controls,
        hessian_ridge=float(args.hessian_ridge),
    )
    scaling = compute_tfidf_scaling(
        df,
        feature_cols,
        stats,
        s_grid=s_grid,
        replications=args.replications,
        seed=args.seed,
        train_pool_size=args.train_pool_size,
        validation_stop_size=args.validation_stop_size,
        validation_scale_size=args.validation_scale_size,
        alphas=alphas,
        max_tfidf_features=args.max_tfidf_features,
        min_df=args.min_df,
        if_weight_clip_quantile=args.if_weight_clip_quantile,
    )
    budget = build_budget_win_table(scaling, summary, budgets)
    scaling_fit = fit_power_law_by_feature(scaling)
    ranked = build_ranked_shortlist(summary, budget)

    df.to_csv(output_dir / "wine_screening_frame.csv", index=False)
    summary.to_csv(output_dir / "candidate_summary.csv", index=False)
    scaling.to_csv(output_dir / "tfidf_scaling_by_feature_s.csv", index=False)
    budget.to_csv(output_dir / "budget_win_table.csv", index=False)
    scaling_fit.to_csv(output_dir / "scaling_fit_table.csv", index=False)
    ranked.to_csv(output_dir / "ranked_shortlist.csv", index=False)
    write_json(output_dir / "data_summary.json", data_summary)
    write_report(output_dir, data_summary, ranked, budget, scaling_fit)
    print(f"wrote {output_dir}", flush=True)
    print(
        ranked.head(10)
        .loc[
            :,
            [
                c
                for c in [
                    "feature",
                    "family",
                    "beta_raw_points",
                    "beta_raw_unit_points",
                    "mse",
                    "mse_stop_ifvar",
                    "ifvar_exact",
                    "ifvar_weighted",
                    "ifvar_beats_mse",
                ]
                if c in ranked.columns
            ],
        ]
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
