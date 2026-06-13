from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.estimators.mean_ppi import attach_truth_metrics, ppi_estimate, ppi_plus_estimate, sample_mean_estimate


LABELED_ROLES = {"train", "validation", "correction"}


def summarize_prediction_file(population: pd.DataFrame, predictions: pd.DataFrame) -> list[dict]:
    true_mean = float(population["points"].mean())
    labeled = predictions[predictions["split_role"].isin(LABELED_ROLES)]
    correction = predictions[predictions["split_role"] == "correction"]
    unlabeled = predictions[predictions["split_role"] == "unlabeled"]
    if correction.empty or unlabeled.empty:
        raise ValueError("prediction file must contain correction and unlabeled rows")

    rows: list[dict] = []
    rows.append(attach_truth_metrics(sample_mean_estimate(labeled["y_true"], "sample_mean").as_dict(), true_mean))

    method = str(predictions["method"].iloc[0])
    y_c = correction["y_true"].to_numpy()
    f_c = correction["pred_mean"].to_numpy()
    f_u = unlabeled["pred_mean"].to_numpy()
    rows.append(attach_truth_metrics(ppi_estimate(y_c, f_c, f_u, method=f"{method}+ppi").as_dict(), true_mean))
    rows.append(attach_truth_metrics(ppi_plus_estimate(y_c, f_c, f_u, method=f"{method}+ppi_plus").as_dict(), true_mean))

    sample_var = next(row["estimated_variance"] for row in rows if row["method"] == "sample_mean")
    for row in rows:
        row["sample_savings"] = 1.0 - float(row["estimated_variance"]) / sample_var if sample_var > 0 else 0.0
    return rows


def summarize(population_path: str | Path, prediction_paths: list[str | Path], output_dir: str | Path) -> pd.DataFrame:
    population = pd.read_csv(population_path)
    all_rows: list[dict] = []
    for pred_path in prediction_paths:
        predictions = pd.read_parquet(pred_path)
        rows = summarize_prediction_file(population, predictions)
        for row in rows:
            row["prediction_file"] = str(pred_path)
        all_rows.extend(rows)
    metrics = pd.DataFrame(all_rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    _write_quick_figure(metrics, out / "estimated_variance.png")
    return metrics


def _write_quick_figure(metrics: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_df = metrics.drop_duplicates(["method", "prediction_file"])
    ax.bar(plot_df["method"], plot_df["estimated_variance"])
    ax.set_ylabel("Estimated variance")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Wine FT+PPI prediction files.")
    parser.add_argument("--population", required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = summarize(args.population, args.predictions, args.output_dir)
    print(metrics)


if __name__ == "__main__":
    main()

