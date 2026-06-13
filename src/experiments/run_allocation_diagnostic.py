from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

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
    budget = int(config["budget"])
    ratios = [float(x) for x in config.get("allocation_ratios", [])]
    if not ratios:
        raise ValueError("config must include allocation_ratios")

    validation_size = int(config.get("validation_size", max(1, round(0.1 * budget))))
    output_root = Path(config.get("output_dir", "artifacts/allocation_diagnostic"))
    runs = []
    for ratio in ratios:
        train_size = allocation_train_size(budget, ratio)
        if train_size + validation_size > budget:
            raise ValueError(
                f"train_size + validation_size exceeds budget for ratio={ratio}: "
                f"{train_size} + {validation_size} > {budget}"
            )
        tag = f"s{train_size:04d}_v{validation_size:04d}"
        run_cfg = dict(config)
        run_cfg["output_dir"] = str(output_root / tag)
        run_cfg["train_size"] = train_size
        run_cfg["validation_size"] = validation_size
        run_cfg["allocation_ratio"] = ratio
        run_cfg["allocation_tag"] = tag
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
