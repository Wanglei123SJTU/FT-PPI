from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analysis.summarize import summarize
from src.estimators.mean_ppi import residual_variance_value
from src.train.train_regression import run_training


def allocation_train_size(budget: int, ratio: float) -> int:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if ratio <= 0 or ratio >= 1:
        raise ValueError("allocation ratios must be in (0, 1)")
    train_size = int(round(budget * ratio))
    if train_size <= 0:
        raise ValueError("allocation ratio gives zero training rows")
    return train_size


def build_allocation_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    budget_values = config.get("budgets")
    if budget_values is None:
        if "budget" not in config:
            raise ValueError("config must include budget or budgets")
        budget_values = [config["budget"]]
    budgets = [int(x) for x in budget_values]
    if not budgets:
        raise ValueError("config must include budget or budgets")
    multi_budget = "budgets" in config or len(budgets) > 1
    ratios = [float(x) for x in config.get("allocation_ratios", [])]
    if not ratios:
        raise ValueError("config must include allocation_ratios")

    replication_ids = [int(x) for x in config.get("replication_ids", [config.get("replication_id", 0)])]
    multi_replication = "replication_ids" in config or len(replication_ids) > 1
    base_seed = int(config.get("seed", 20260612))
    population_seed = int(config.get("population_seed", base_seed))
    split_seed_base = int(config.get("split_seed", base_seed))

    output_root = Path(config.get("output_dir", "artifacts/allocation_diagnostic"))
    runs = []
    for replication_id in replication_ids:
        for budget in budgets:
            if budget <= 0:
                raise ValueError("budget must be positive")
            validation_size = int(config.get("validation_size", max(1, round(0.1 * budget))))
            for ratio in ratios:
                train_size = allocation_train_size(budget, ratio)
                if train_size + validation_size > budget:
                    raise ValueError(
                        f"train_size + validation_size exceeds budget for ratio={ratio}: "
                        f"{train_size} + {validation_size} > {budget}"
                    )
                tag = f"s{train_size:04d}_v{validation_size:04d}"
                if multi_budget:
                    tag = f"B{budget:04d}_{tag}"
                if multi_replication:
                    tag = f"r{replication_id:03d}_{tag}"
                run_cfg = dict(config)
                run_cfg["budget"] = budget
                run_cfg["output_dir"] = str(output_root / tag)
                run_cfg["train_size"] = train_size
                run_cfg["validation_size"] = validation_size
                run_cfg["allocation_ratio"] = ratio
                run_cfg["allocation_tag"] = tag
                run_cfg["replication_id"] = replication_id
                run_cfg["population_seed"] = population_seed
                run_cfg["split_seed"] = split_seed_base + replication_id
                runs.append(run_cfg)
    return runs


def _loss_from_path(prediction_file: str) -> str:
    parent = Path(prediction_file).parent.name
    return parent if parent in {"mse", "var"} else ""


def prediction_residual_variances(prediction_paths: list[Path]) -> dict[str, float]:
    out: dict[str, float] = {}
    for path in prediction_paths:
        predictions = pd.read_parquet(path)
        correction = predictions[predictions["split_role"] == "correction"]
        if correction.empty:
            raise ValueError(f"{path} has no correction rows")
        loss = str(predictions["loss"].iloc[0]) if "loss" in predictions.columns else path.parent.name
        out[loss] = residual_variance_value(correction["y_true"], correction["pred_mean"])
    return out


def add_diagnostic_columns(metrics: pd.DataFrame, run_cfg: dict[str, Any], residual_vars: dict[str, float]) -> pd.DataFrame:
    out = metrics.copy()
    out["allocation_ratio"] = float(run_cfg["allocation_ratio"])
    out["train_size"] = int(run_cfg["train_size"])
    out["validation_size"] = int(run_cfg["validation_size"])
    out["budget"] = int(run_cfg["budget"])
    out["population_size"] = int(run_cfg["population_size"])
    out["replication_id"] = int(run_cfg.get("replication_id", 0))
    out["loss"] = out["prediction_file"].map(_loss_from_path)
    out.loc[out["method"].astype(str) == "sample_mean", "loss"] = "sample_mean"
    out["residual_variance"] = out["loss"].map(residual_vars)
    out["estimated_estimator_variance"] = out["estimated_variance"]
    return out


def _write_quick_figures(metrics: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_df = metrics.drop_duplicates(["allocation_ratio", "method", "loss", "prediction_file"])
    fig, ax = plt.subplots(figsize=(9, 4))
    for method, group in plot_df.groupby("method"):
        ax.plot(group["train_size"], group["estimated_estimator_variance"], marker="o", label=method)
    ax.set_xlabel("Training rows")
    ax.set_ylabel("Estimated estimator variance")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "estimated_variance_by_allocation.png")
    plt.close(fig)

    residual_df = plot_df[plot_df["loss"].isin(["mse", "var"])].drop_duplicates(["train_size", "loss"])
    if not residual_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for loss, group in residual_df.groupby("loss"):
            ax.plot(group["train_size"], group["residual_variance"], marker="o", label=loss)
        ax.set_xlabel("Training rows")
        ax.set_ylabel("Correction residual variance")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "residual_variance_by_allocation.png")
        plt.close(fig)


def _mean_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def summarize_allocation_curve(metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["budget", "allocation_ratio", "train_size", "validation_size", "loss", "method"]
    rows = []
    for keys, group in metrics.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_replications"] = int(group["replication_id"].nunique())
        row["mean_estimated_variance"] = _mean_or_nan(group["estimated_estimator_variance"])
        row["sd_estimated_variance"] = float(group["estimated_estimator_variance"].std(ddof=1)) if len(group) > 1 else 0.0
        row["mean_ci_length"] = _mean_or_nan(group["ci_length"])
        row["mean_residual_variance"] = _mean_or_nan(group["residual_variance"])
        row["mean_sample_savings"] = _mean_or_nan(group["sample_savings"])
        row["mean_bias"] = _mean_or_nan(group["bias"])
        row["mean_rmse"] = _mean_or_nan(group["rmse"])
        rows.append(row)
    curve = pd.DataFrame(rows).sort_values(["budget", "method", "loss", "train_size"]).reset_index(drop=True)
    sample_ref = (
        curve[curve["method"] == "sample_mean"][["budget", "mean_estimated_variance"]]
        .drop_duplicates("budget")
        .rename(columns={"mean_estimated_variance": "sample_mean_variance"})
    )
    curve = curve.merge(sample_ref, on="budget", how="left")
    curve["normalized_estimated_variance"] = curve["mean_estimated_variance"] / curve["sample_mean_variance"]
    return curve


def summarize_oracle_allocations(curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = curve[curve["method"].astype(str).str.contains("ppi_plus", regex=False)].copy()
    for (budget, method, loss), group in target.groupby(["budget", "method", "loss"], dropna=False):
        group = group.sort_values("mean_estimated_variance")
        best = group.iloc[0]
        sample_var = float(best["sample_mean_variance"])
        rows.append(
            {
                "budget": int(budget),
                "method": method,
                "loss": loss,
                "oracle_train_size": int(best["train_size"]),
                "oracle_allocation_ratio": float(best["allocation_ratio"]),
                "oracle_estimated_variance": float(best["mean_estimated_variance"]),
                "oracle_normalized_variance": float(best["mean_estimated_variance"]) / sample_var if sample_var > 0 else np.nan,
                "oracle_ci_length": float(best["mean_ci_length"]),
                "n_replications": int(best["n_replications"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["budget", "method", "loss"]).reset_index(drop=True)


def _fit_power_law(train_sizes: np.ndarray, residual_variances: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(train_sizes) & np.isfinite(residual_variances) & (train_sizes > 0) & (residual_variances > 0)
    x = train_sizes[mask].astype(float)
    y = residual_variances[mask].astype(float)
    if len(x) < 3:
        return {"a": np.nan, "alpha": np.nan, "b": np.nan, "r2": np.nan}

    best = None
    upper = float(min(y) * 0.95)
    for b in np.linspace(0.0, upper, 80):
        adjusted = y - b
        if np.any(adjusted <= 0):
            continue
        slope, intercept = np.polyfit(np.log(x), np.log(adjusted), deg=1)
        pred = np.exp(intercept) * x ** slope + b
        sse = float(np.sum((y - pred) ** 2))
        if best is None or sse < best["sse"]:
            best = {"a": float(np.exp(intercept)), "alpha": float(-slope), "b": float(b), "sse": sse, "pred": pred}
    if best is None:
        return {"a": np.nan, "alpha": np.nan, "b": np.nan, "r2": np.nan}
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - best["sse"] / sst if sst > 0 else np.nan
    return {"a": best["a"], "alpha": best["alpha"], "b": best["b"], "r2": float(r2)}


def _power_law_predict(train_sizes: np.ndarray, fit: dict[str, float]) -> np.ndarray:
    if not np.isfinite([fit["a"], fit["alpha"], fit["b"]]).all():
        return np.full_like(train_sizes, np.nan, dtype=float)
    x = train_sizes.astype(float)
    return fit["a"] * x ** (-fit["alpha"]) + fit["b"]


def summarize_scaling_laws(curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    loso_rows = []
    source = curve[curve["loss"].isin(["mse", "var"])].drop_duplicates(["budget", "loss", "train_size"]).copy()
    for (budget, loss), group in source.groupby(["budget", "loss"], dropna=False):
        group = group.sort_values("train_size")
        x = group["train_size"].to_numpy(dtype=float)
        y = group["mean_residual_variance"].to_numpy(dtype=float)
        fit = _fit_power_law(x, y)
        pred = _power_law_predict(x, fit)
        for train_size, observed, predicted in zip(x, y, pred):
            loso_group = group[group["train_size"] != train_size]
            loso_fit = _fit_power_law(
                loso_group["train_size"].to_numpy(dtype=float),
                loso_group["mean_residual_variance"].to_numpy(dtype=float),
            )
            loso_pred = float(_power_law_predict(np.array([train_size], dtype=float), loso_fit)[0])
            loso_rows.append(
                {
                    "budget": int(budget),
                    "loss": loss,
                    "heldout_train_size": int(train_size),
                    "observed_residual_variance": float(observed),
                    "fitted_residual_variance": float(predicted),
                    "loso_predicted_residual_variance": loso_pred,
                    "loso_relative_error": abs(float(observed) - loso_pred) / float(observed) if observed > 0 and np.isfinite(loso_pred) else np.nan,
                }
            )
        feasible = np.arange(max(1, int(x.min())), int(budget - group["validation_size"].iloc[0]), dtype=float)
        fitted_var = _power_law_predict(feasible, fit)
        correction = budget - float(group["validation_size"].iloc[0]) - feasible
        objective = np.where(correction > 0, fitted_var / correction, np.nan)
        best_idx = int(np.nanargmin(objective)) if np.isfinite(objective).any() else 0
        rows.append(
            {
                "budget": int(budget),
                "loss": loss,
                "a": fit["a"],
                "alpha": fit["alpha"],
                "b": fit["b"],
                "r2": fit["r2"],
                "predicted_train_size": int(feasible[best_idx]) if len(feasible) else np.nan,
                "predicted_allocation_ratio": float(feasible[best_idx] / budget) if len(feasible) else np.nan,
                "mean_loso_relative_error": _mean_or_nan(pd.Series([row["loso_relative_error"] for row in loso_rows if row["budget"] == int(budget) and row["loss"] == loss])),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(loso_rows)


def write_summary_tables(metrics: pd.DataFrame, summary_dir: Path) -> None:
    curve = summarize_allocation_curve(metrics)
    oracle = summarize_oracle_allocations(curve)
    scaling, loso = summarize_scaling_laws(curve)
    curve.to_csv(summary_dir / "allocation_curve.csv", index=False)
    oracle.to_csv(summary_dir / "oracle_allocations.csv", index=False)
    scaling.to_csv(summary_dir / "scaling_law_fit.csv", index=False)
    loso.to_csv(summary_dir / "scaling_law_loso.csv", index=False)


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_allocation_diagnostic(config: dict[str, Any]) -> pd.DataFrame:
    output_root = Path(config.get("output_dir", "artifacts/allocation_diagnostic"))
    output_root.mkdir(parents=True, exist_ok=True)
    losses = [str(x) for x in config.get("losses", ["mse", "var"])]
    unsupported = set(losses) - {"mse", "var"}
    if unsupported:
        raise ValueError(f"unsupported losses: {sorted(unsupported)}")

    all_metrics = []
    for run_cfg in build_allocation_runs(config):
        run_dir = Path(run_cfg["output_dir"])
        print(
            "allocation_run_start",
            f"tag={run_cfg['allocation_tag']}",
            f"train_size={run_cfg['train_size']}",
            f"validation_size={run_cfg['validation_size']}",
            f"budget={run_cfg['budget']}",
            flush=True,
        )
        for loss in losses:
            print(f"training loss={loss} output_dir={run_dir}", flush=True)
            run_training(run_cfg, loss)
            _release_cuda_cache()

        prediction_paths = [run_dir / loss / "predictions.parquet" for loss in losses]
        metrics = summarize(
            population_path=run_dir / losses[0] / "population.csv",
            prediction_paths=prediction_paths,
            output_dir=run_dir / "summary",
        )
        residual_vars = prediction_residual_variances(prediction_paths)
        metrics = add_diagnostic_columns(metrics, run_cfg, residual_vars)
        metrics.to_csv(run_dir / "summary" / "metrics.csv", index=False)
        all_metrics.append(metrics)
        print(f"allocation_run_done tag={run_cfg['allocation_tag']}", flush=True)

    combined = pd.concat(all_metrics, ignore_index=True)
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(summary_dir / "allocation_metrics.csv", index=False)
    diagnostic_cols = [
        "allocation_ratio",
        "train_size",
        "validation_size",
        "budget",
        "loss",
        "method",
        "residual_variance",
        "estimated_estimator_variance",
        "standard_error",
        "ci_length",
        "sample_savings",
        "bias",
        "rmse",
        "lambda",
    ]
    existing_cols = [col for col in diagnostic_cols if col in combined.columns]
    combined[existing_cols].to_csv(summary_dir / "diagnostic_summary.csv", index=False)
    write_summary_tables(combined, summary_dir)
    _write_quick_figures(combined, summary_dir)
    print(combined[existing_cols], flush=True)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the small Wine allocation diagnostic.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_allocation_diagnostic(config)


if __name__ == "__main__":
    main()
