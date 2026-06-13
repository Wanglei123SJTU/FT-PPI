from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


TAG_RE = re.compile(r"r(?P<replication_id>\d+)_s(?P<train_size>\d+)_v(?P<validation_size>\d+)")


def _safe_corr(y_true: pd.Series, pred: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(pred, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) < 2 or np.std(y) <= 0 or np.std(p) <= 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def _residual_variance(y_true: pd.Series, pred: pd.Series) -> float:
    residual = pd.to_numeric(y_true, errors="coerce") - pd.to_numeric(pred, errors="coerce")
    residual = residual.dropna()
    if residual.empty:
        return float("nan")
    centered = residual - residual.mean()
    return float(np.mean(centered.to_numpy(dtype=float) ** 2))


def _parse_prediction_path(path: Path) -> dict[str, int | str]:
    tag = path.parents[1].name
    match = TAG_RE.fullmatch(tag)
    if not match:
        raise ValueError(f"prediction path does not include allocation tag: {path}")
    return {
        "allocation_tag": tag,
        "replication_id": int(match.group("replication_id")),
        "train_size": int(match.group("train_size")),
        "validation_size": int(match.group("validation_size")),
        "loss": path.parent.name,
    }


def summarize_prediction_quality(predictions: pd.DataFrame, metadata: dict[str, int | str]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    role_masks = {
        "train": predictions["split_role"] == "train",
        "validation": predictions["split_role"] == "validation",
        "correction": predictions["split_role"] == "correction",
        "unlabeled": predictions["split_role"] == "unlabeled",
        "population": pd.Series(True, index=predictions.index),
    }
    for role, mask in role_masks.items():
        subset = predictions[mask]
        if subset.empty:
            continue
        residual = subset["pred_mean"].astype(float) - subset["y_true"].astype(float)
        rows.append(
            {
                **metadata,
                "role": role,
                "n": int(len(subset)),
                "corr": _safe_corr(subset["y_true"], subset["pred_mean"]),
                "rmse": float(np.sqrt(np.mean(residual.to_numpy(dtype=float) ** 2))),
                "bias": float(residual.mean()),
                "pred_mean": float(subset["pred_mean"].mean()),
                "pred_std": float(subset["pred_mean"].std(ddof=1)) if len(subset) > 1 else 0.0,
                "y_mean": float(subset["y_true"].mean()),
                "y_std": float(subset["y_true"].std(ddof=1)) if len(subset) > 1 else 0.0,
                "residual_variance": _residual_variance(subset["y_true"], subset["pred_mean"]),
            }
        )
    return rows


def build_prediction_diagnostics(root_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root_dir)
    rows: list[dict[str, float | int | str]] = []
    for path in sorted(root.glob("r*_s*_v*/*/predictions.parquet")):
        metadata = _parse_prediction_path(path)
        predictions = pd.read_parquet(path)
        rows.extend(summarize_prediction_quality(predictions, metadata))
    per_run = pd.DataFrame(rows)
    if per_run.empty:
        return per_run, per_run

    group_cols = ["loss", "train_size", "role"]
    grouped = per_run.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        n_replications=("replication_id", "nunique"),
        mean_corr=("corr", "mean"),
        sd_corr=("corr", "std"),
        mean_rmse=("rmse", "mean"),
        mean_bias=("bias", "mean"),
        mean_pred_mean=("pred_mean", "mean"),
        mean_pred_std=("pred_std", "mean"),
        mean_y_std=("y_std", "mean"),
        mean_residual_variance=("residual_variance", "mean"),
    ).reset_index()
    summary["sd_corr"] = summary["sd_corr"].fillna(0.0)
    return per_run, summary.sort_values(["loss", "role", "train_size"]).reset_index(drop=True)


def write_prediction_diagnostics(root_dir: str | Path, output_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run, summary = build_prediction_diagnostics(root_dir)
    out = Path(output_dir) if output_dir is not None else Path(root_dir) / "summary"
    out.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(out / "prediction_diagnostics_by_run.csv", index=False)
    summary.to_csv(out / "prediction_diagnostics.csv", index=False)
    return per_run, summary
