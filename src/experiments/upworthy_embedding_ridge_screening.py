from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from src.experiments.upworthy_embedding_mlp_screening import (
    DEFAULT_BUDGETS,
    DEFAULT_CONTROLS,
    DEFAULT_S_GRID,
    DEFAULT_TARGETS,
    budget_rows_for_prediction,
    eligible_controls,
    evaluate_predictions,
    fit_eval_ols,
    if_weights_for_frame,
    load_npz,
    pair_embeddings,
    parse_csv_list,
    parse_int_list,
    standardize_embeddings,
    write_json,
)
from src.formatting import dataframe_to_markdown


@dataclass(frozen=True)
class RidgeMethod:
    method: str
    sample_weight: str
    stop_metric: str


METHODS = [
    RidgeMethod("ridge_mse_stop_mse", "uniform", "mse"),
    RidgeMethod("ridge_mse_stop_ifvar", "uniform", "ifvar"),
    RidgeMethod("ridge_ifwls_stop_ifvar", "if_weight_sq", "ifvar"),
]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def fit_ridge_path(
    x: np.ndarray,
    y: np.ndarray,
    if_weights: np.ndarray,
    *,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    method: RidgeMethod,
    alphas: list[float],
) -> tuple[np.ndarray, dict[str, float]]:
    best: tuple[float, float, Ridge, float, float] | None = None
    train_weight = None
    if method.sample_weight == "if_weight_sq":
        train_weight = np.square(if_weights[train_indices])
        train_weight = train_weight / max(float(np.mean(train_weight)), 1e-12)
    elif method.sample_weight != "uniform":
        raise ValueError(f"unknown sample_weight: {method.sample_weight}")

    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(x[train_indices], y[train_indices], sample_weight=train_weight)
        pred = model.predict(x[validation_indices])
        residual = y[validation_indices] - pred
        mse = mean_squared_error(y[validation_indices], pred)
        if_resid = if_weights[validation_indices] * residual
        ifvar = float(np.var(if_resid, ddof=0))
        metric = mse if method.stop_metric == "mse" else ifvar
        if best is None or metric < best[1]:
            best = (float(alpha), float(metric), model, float(mse), ifvar)
    assert best is not None
    alpha, metric, model, val_mse, val_ifvar = best
    return model.predict(x).astype(np.float32), {
        "alpha": alpha,
        "best_validation_metric": metric,
        "validation_mse": val_mse,
        "validation_ifvar": val_ifvar,
    }


def select_train_validation(
    train_pool_indices: np.ndarray,
    *,
    s: int,
    seed: int,
    min_validation: int,
) -> tuple[np.ndarray, np.ndarray]:
    if s + min_validation > len(train_pool_indices):
        raise ValueError(f"s={s} plus min_validation={min_validation} exceeds train pool {len(train_pool_indices)}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_pool_indices)
    return np.sort(perm[:s]), np.sort(perm[s:])


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = load_npz(args.embedding_npz)
    frame = pd.read_csv(args.sample_csv)
    metadata = pd.read_csv(args.feature_metadata_csv) if args.feature_metadata_csv and Path(args.feature_metadata_csv).exists() else None
    embeddings_raw = pair_embeddings(cache, args.representation)
    train_pool_indices = np.flatnonzero(frame["screen_split"].astype(str).to_numpy() == "train_pool")
    evaluation_indices = np.flatnonzero(frame["screen_split"].astype(str).to_numpy() == "evaluation")
    embeddings, _, _ = standardize_embeddings(embeddings_raw, train_pool_indices)
    y = frame["y_logit_ctr_diff"].astype(float).to_numpy()
    y_mean = float(y[train_pool_indices].mean()) if args.standardize_y else 0.0
    y_sd = float(y[train_pool_indices].std(ddof=0)) if args.standardize_y else 1.0
    if not np.isfinite(y_sd) or y_sd <= 1e-8:
        y_sd = 1.0
    y_scaled = ((y - y_mean) / y_sd).astype(np.float32)

    targets = parse_csv_list(args.targets)
    controls = parse_csv_list(args.controls)
    s_grid = parse_int_list(args.s_grid)
    budgets = parse_int_list(args.budgets)
    alphas = parse_float_list(args.alphas)
    eval_frame = frame.iloc[evaluation_indices].copy()
    y_eval = y[evaluation_indices]

    results: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    for target in targets:
        selected_controls = eligible_controls(eval_frame, metadata, target=target, controls=controls)
        beta, hessian, ols_residual, x_columns = fit_eval_ols(eval_frame, target=target, controls=selected_controls)
        weights_all = if_weights_for_frame(
            frame,
            columns=x_columns,
            hessian=hessian,
            target=target,
            ridge=args.hessian_ridge,
        )
        weights_eval = weights_all[evaluation_indices]
        zero_metrics = evaluate_predictions(y_eval, np.zeros_like(y_eval), weights_eval)
        direct_ols_ifvar = float(np.var(weights_eval * ols_residual, ddof=0))
        target_rows.append(
            {
                "target": target,
                "controls": ",".join(selected_controls),
                "beta_eval": float(beta[1]),
                "direct_ols_ifvar": direct_ols_ifvar,
                "zero_surrogate_ifvar": zero_metrics["ifvar"],
                "zero_surrogate_mse": zero_metrics["mse"],
                "hessian_condition": float(np.linalg.cond(hessian)),
                "if_weight_p99_abs": float(np.quantile(np.abs(weights_eval), 0.99)),
                "if_weight_max_abs": float(np.max(np.abs(weights_eval))),
            }
        )
        results.append(
            {
                "target": target,
                "method": "zero_surrogate",
                "replication": -1,
                "s": 0,
                "alpha": np.nan,
                "mse": zero_metrics["mse"],
                "ifvar": zero_metrics["ifvar"],
                "ifvar_ratio_vs_zero": 1.0,
                "ifvar_ratio_vs_direct_ols": zero_metrics["ifvar"] / direct_ols_ifvar if direct_ols_ifvar > 0 else np.nan,
            }
        )
        budget_rows.extend(
            budget_rows_for_prediction(
                target=target,
                method="zero_surrogate",
                s=0,
                ifvar=zero_metrics["ifvar"],
                direct_ols_ifvar=direct_ols_ifvar,
                zero_ifvar=zero_metrics["ifvar"],
                budgets=budgets,
            )
        )

        for s in s_grid:
            if s <= 0:
                continue
            if s + args.min_validation > len(train_pool_indices):
                print(f"skip target={target} s={s}: not enough validation", flush=True)
                continue
            for rep in range(args.replications):
                train_idx, val_idx = select_train_validation(
                    train_pool_indices,
                    s=s,
                    seed=args.seed + 1009 * rep + 17 * s,
                    min_validation=args.min_validation,
                )
                for method in METHODS:
                    pred_scaled, diagnostics = fit_ridge_path(
                        embeddings,
                        y_scaled,
                        weights_all,
                        train_indices=train_idx,
                        validation_indices=val_idx,
                        method=method,
                        alphas=alphas,
                    )
                    pred = pred_scaled * y_sd + y_mean
                    metrics = evaluate_predictions(y_eval, pred[evaluation_indices], weights_eval)
                    results.append(
                        {
                            "target": target,
                            "method": method.method,
                            "replication": rep,
                            "s": int(s),
                            "alpha": diagnostics["alpha"],
                            "mse": metrics["mse"],
                            "ifvar": metrics["ifvar"],
                            "ifvar_ratio_vs_zero": metrics["ifvar"] / zero_metrics["ifvar"] if zero_metrics["ifvar"] > 0 else np.nan,
                            "ifvar_ratio_vs_direct_ols": metrics["ifvar"] / direct_ols_ifvar if direct_ols_ifvar > 0 else np.nan,
                            "validation_mse": diagnostics["validation_mse"],
                            "validation_ifvar": diagnostics["validation_ifvar"],
                        }
                    )
                    budget_rows.extend(
                        budget_rows_for_prediction(
                            target=target,
                            method=method.method,
                            s=s,
                            ifvar=metrics["ifvar"],
                            direct_ols_ifvar=direct_ols_ifvar,
                            zero_ifvar=zero_metrics["ifvar"],
                            budgets=budgets,
                        )
                    )
                    print(
                        f"target={target} method={method.method} rep={rep} s={s} "
                        f"ratio={metrics['ifvar'] / zero_metrics['ifvar']:.4f}",
                        flush=True,
                    )
    return pd.DataFrame(results), pd.DataFrame(budget_rows), pd.DataFrame(target_rows)


def write_outputs(
    output_dir: Path,
    results: pd.DataFrame,
    budgets: pd.DataFrame,
    targets: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        results.loc[results["s"] > 0]
        .groupby(["target", "method", "s"], as_index=False)
        .agg(
            mean_ifvar=("ifvar", "mean"),
            se_ifvar=("ifvar", lambda x: float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0),
            mean_ifvar_ratio_vs_zero=("ifvar_ratio_vs_zero", "mean"),
            mean_ifvar_ratio_vs_direct_ols=("ifvar_ratio_vs_direct_ols", "mean"),
            mean_mse=("mse", "mean"),
            median_alpha=("alpha", "median"),
        )
    )
    budget_summary = (
        budgets.groupby(["target", "method", "s", "budget"], as_index=False)
        .agg(
            mean_ratio_vs_direct_ols=("ratio_vs_direct_ols", "mean"),
            win_rate_direct_ols=("beats_direct_ols", "mean"),
            mean_ratio_vs_zero_surrogate=("ratio_vs_zero_surrogate", "mean"),
        )
        .sort_values(["target", "budget", "mean_ratio_vs_direct_ols"])
    )
    best = summary.sort_values(["target", "mean_ifvar_ratio_vs_zero"]).groupby("target", as_index=False).head(5)
    results.to_csv(output_dir / "embedding_ridge_results.csv", index=False)
    summary.to_csv(output_dir / "embedding_ridge_summary.csv", index=False)
    budgets.to_csv(output_dir / "budget_win_rows.csv", index=False)
    budget_summary.to_csv(output_dir / "budget_win_summary.csv", index=False)
    targets.to_csv(output_dir / "target_diagnostics.csv", index=False)
    write_json(
        output_dir / "run_metadata.json",
        {
            "embedding_npz": str(args.embedding_npz),
            "sample_csv": str(args.sample_csv),
            "targets": parse_csv_list(args.targets),
            "controls": parse_csv_list(args.controls),
            "s_grid": parse_int_list(args.s_grid),
            "budgets": parse_int_list(args.budgets),
            "replications": int(args.replications),
            "alphas": parse_float_list(args.alphas),
            "representation": args.representation,
        },
    )
    with open(output_dir / "screening_report.md", "w", encoding="utf-8") as f:
        f.write("# Upworthy OpenAI Embedding Ridge Screening\n\n")
        f.write("This is a fast semantic-surrogate screen using cached OpenAI embeddings and closed-form ridge heads.\n\n")
        f.write("## Target Diagnostics\n\n")
        f.write(dataframe_to_markdown(targets.round(4)))
        f.write("\n\n## Best Variance Reductions\n\n")
        f.write(dataframe_to_markdown(best.round(4)))
        f.write("\n\n## Budget Win Summary: Best Rows\n\n")
        f.write(dataframe_to_markdown(budget_summary.groupby(["target", "budget"], as_index=False).head(1).round(4)))
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast Upworthy semantic screening with cached OpenAI embeddings and ridge heads.")
    parser.add_argument("--embedding-npz", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--feature-metadata-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/upworthy_m_estimation/openai_embedding_ridge_screening"))
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--controls", default=",".join(DEFAULT_CONTROLS))
    parser.add_argument("--s-grid", default=",".join(str(item) for item in DEFAULT_S_GRID))
    parser.add_argument("--budgets", default=",".join(str(item) for item in DEFAULT_BUDGETS))
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100,1000,10000")
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--min-validation", type=int, default=500)
    parser.add_argument("--representation", choices=["diff", "diff_abs"], default="diff")
    parser.add_argument("--hessian-ridge", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--no-standardize-y", dest="standardize_y", action="store_false")
    parser.set_defaults(standardize_y=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, budgets, targets = run(args)
    write_outputs(args.output_dir, results, budgets, targets, args)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
