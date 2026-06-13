from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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
        pred = np.exp(intercept) * x**slope + b
        sse = float(np.sum((y - pred) ** 2))
        if best is None or sse < best["sse"]:
            best = {"a": float(np.exp(intercept)), "alpha": float(-slope), "b": float(b), "sse": sse}
    if best is None:
        return {"a": np.nan, "alpha": np.nan, "b": np.nan, "r2": np.nan}

    pred = best["a"] * x ** (-best["alpha"]) + best["b"]
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - best["sse"] / sst if sst > 0 else np.nan
    return {"a": best["a"], "alpha": best["alpha"], "b": best["b"], "r2": float(r2)}


def _predict_power_law(train_sizes: np.ndarray, fit: dict[str, float]) -> np.ndarray:
    values = np.array([fit["a"], fit["alpha"], fit["b"]], dtype=float)
    if not np.isfinite(values).all():
        return np.full_like(train_sizes, np.nan, dtype=float)
    x = train_sizes.astype(float)
    return fit["a"] * x ** (-fit["alpha"]) + fit["b"]


def _mean_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else float("nan")


def _rampup_points(group: pd.DataFrame, rampup_points: int, max_train_size: int | None) -> pd.DataFrame:
    ordered = group.sort_values("train_size").drop_duplicates("train_size")
    if max_train_size is not None:
        ordered = ordered[ordered["train_size"] <= max_train_size]
    return ordered.head(rampup_points)


def summarize_rampup_allocation_regret(
    curve: pd.DataFrame,
    rampup_points: int = 3,
    max_train_size: int | None = None,
) -> pd.DataFrame:
    if rampup_points < 3:
        raise ValueError("rampup_points must be at least 3 for the power-law fit")
    required = {
        "budget",
        "allocation_ratio",
        "train_size",
        "validation_size",
        "loss",
        "method",
        "mean_residual_variance",
        "mean_estimated_variance",
        "normalized_estimated_variance",
        "sample_mean_variance",
    }
    missing = required - set(curve.columns)
    if missing:
        raise ValueError(f"allocation curve missing columns: {sorted(missing)}")

    rows: list[dict] = []
    target = curve[curve["method"].astype(str).str.contains("ppi_plus", regex=False)].copy()
    for (budget, method, loss), method_group in target.groupby(["budget", "method", "loss"], dropna=False):
        grid = method_group.sort_values("train_size").drop_duplicates("train_size")
        residual_source = grid[["train_size", "validation_size", "mean_residual_variance"]].dropna()
        ramp = _rampup_points(residual_source, rampup_points, max_train_size)
        fit = _fit_power_law(
            ramp["train_size"].to_numpy(dtype=float),
            ramp["mean_residual_variance"].to_numpy(dtype=float),
        )

        train_sizes = grid["train_size"].to_numpy(dtype=float)
        fitted_residual = _predict_power_law(train_sizes, fit)
        correction = float(budget) - grid["validation_size"].to_numpy(dtype=float) - train_sizes
        objective = np.where(correction > 0, fitted_residual / correction, np.nan)
        selected_pos = int(np.nanargmin(objective)) if np.isfinite(objective).any() else 0
        selected = grid.iloc[selected_pos]
        oracle = grid.sort_values("mean_estimated_variance").iloc[0]
        oracle_variance = float(oracle["mean_estimated_variance"])
        selected_variance = float(selected["mean_estimated_variance"])
        sample_mean_variance = float(selected["sample_mean_variance"])

        rows.append(
            {
                "budget": int(budget),
                "method": method,
                "loss": loss,
                "rampup_train_sizes": ",".join(str(int(x)) for x in ramp["train_size"]),
                "n_rampup_points": int(len(ramp)),
                "fit_a": fit["a"],
                "fit_alpha": fit["alpha"],
                "fit_b": fit["b"],
                "fit_r2": fit["r2"],
                "selected_train_size": int(selected["train_size"]),
                "selected_allocation_ratio": float(selected["allocation_ratio"]),
                "selected_estimated_variance": selected_variance,
                "selected_normalized_variance": float(selected["normalized_estimated_variance"]),
                "selected_sample_savings": 1.0 - selected_variance / sample_mean_variance
                if sample_mean_variance > 0
                else np.nan,
                "oracle_train_size": int(oracle["train_size"]),
                "oracle_allocation_ratio": float(oracle["allocation_ratio"]),
                "oracle_estimated_variance": oracle_variance,
                "oracle_normalized_variance": float(oracle["normalized_estimated_variance"]),
                "oracle_sample_savings": 1.0 - oracle_variance / sample_mean_variance
                if sample_mean_variance > 0
                else np.nan,
                "relative_regret": selected_variance / oracle_variance - 1.0 if oracle_variance > 0 else np.nan,
                "beats_sample_mean": bool(selected_variance < sample_mean_variance),
                "sample_mean_variance": sample_mean_variance,
                "mean_rampup_residual_variance": _mean_or_nan(ramp["mean_residual_variance"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["budget", "method", "loss"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize deployable ramp-up allocation regret from an allocation grid.")
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--rampup-points", type=int, default=3)
    parser.add_argument("--max-train-size", type=int, default=None)
    parser.add_argument("--output-name", default="rampup_allocation_regret.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    curve_path = summary_dir / "allocation_curve.csv"
    curve = pd.read_csv(curve_path)
    regret = summarize_rampup_allocation_regret(
        curve,
        rampup_points=args.rampup_points,
        max_train_size=args.max_train_size,
    )
    output_path = summary_dir / args.output_name
    regret.to_csv(output_path, index=False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print(f"wrote {output_path}")
    print(regret.to_string(index=False))


if __name__ == "__main__":
    main()
