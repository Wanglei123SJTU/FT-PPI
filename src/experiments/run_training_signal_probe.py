from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.experiments.run_allocation_diagnostic import run_allocation_diagnostic


def build_probe_configs(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    output_root = Path(config.get("output_dir", "artifacts/training_signal_probe"))
    base = dict(config.get("base", {}))
    variants = config.get("variants", [])
    if not variants:
        raise ValueError("training signal probe config must include variants")

    runs: list[tuple[str, dict[str, Any]]] = []
    for variant in variants:
        if "name" not in variant:
            raise ValueError("each variant must include a name")
        name = str(variant["name"])
        overrides = {key: value for key, value in variant.items() if key != "name"}
        run_cfg = dict(base)
        run_cfg.update(overrides)
        run_cfg["output_dir"] = str(output_root / name)
        run_cfg.setdefault("resume_completed", bool(config.get("resume_completed", True)))
        run_cfg.setdefault("save_adapter", False)
        runs.append((name, run_cfg))
    return runs


def run_training_signal_probe(config: dict[str, Any]) -> None:
    output_root = Path(config.get("output_dir", "artifacts/training_signal_probe"))
    output_root.mkdir(parents=True, exist_ok=True)

    diagnostic_frames = []
    curve_frames = []
    for name, run_cfg in build_probe_configs(config):
        print(f"training_signal_probe_start variant={name}", flush=True)
        run_allocation_diagnostic(run_cfg)
        summary_dir = Path(run_cfg["output_dir"]) / "summary"

        diagnostics = pd.read_csv(summary_dir / "prediction_diagnostics.csv")
        diagnostics.insert(0, "variant", name)
        diagnostic_frames.append(diagnostics)

        curve = pd.read_csv(summary_dir / "allocation_curve.csv")
        curve.insert(0, "variant", name)
        curve_frames.append(curve)
        print(f"training_signal_probe_done variant={name}", flush=True)

    combined_dir = output_root / "summary"
    combined_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_all = pd.concat(diagnostic_frames, ignore_index=True)
    curve_all = pd.concat(curve_frames, ignore_index=True)
    diagnostics_all.to_csv(combined_dir / "prediction_diagnostics.csv", index=False)
    curve_all.to_csv(combined_dir / "allocation_curve.csv", index=False)

    diagnostic_cols = [
        "variant",
        "loss",
        "train_size",
        "role",
        "mean_corr",
        "mean_rmse",
        "mean_bias",
        "mean_pred_mean",
        "mean_pred_std",
        "mean_residual_variance",
    ]
    print("training signal prediction diagnostics", flush=True)
    print(
        diagnostics_all[diagnostics_all["role"].isin(["population", "correction"])][diagnostic_cols]
        .sort_values(["variant", "role"])
        .to_string(index=False),
        flush=True,
    )

    curve_cols = [
        "variant",
        "method",
        "loss",
        "train_size",
        "mean_estimated_variance",
        "normalized_estimated_variance",
        "mean_sample_savings",
    ]
    ppi_plus = curve_all[curve_all["method"].astype(str).str.contains("ppi_plus", regex=False)]
    print("training signal ppi_plus curve", flush=True)
    print(ppi_plus[curve_cols].sort_values(["variant", "method"]).to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small Wine LoRA training-signal hyperparameter probe.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_training_signal_probe(config)


if __name__ == "__main__":
    main()
