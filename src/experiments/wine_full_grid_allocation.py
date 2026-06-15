from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.experiments.wine_var_scaling_law import (
    Y_COL,
    build_population_split,
    clean_wine,
    configured_methods,
    ids_hash,
    is_cuda_oom,
    method_by_name,
    prediction_metrics,
    split_source,
    subset_by_ids,
    train_once,
    write_json,
)


@dataclass(frozen=True)
class AllocationSplit:
    budget_B: int
    validation_ids: np.ndarray
    labeled_ids: np.ndarray
    effective_ids: np.ndarray
    train_order_ids: np.ndarray
    unlabeled_ids: np.ndarray


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def allocation_seed(config: dict[str, Any], replication_id: int, budget_B: int, salt: int = 0) -> int:
    return int(config.get("seed", 20260613)) + 100_003 * int(replication_id) + 1_009 * int(budget_B) + int(salt)


def validation_size_for_budget(config: dict[str, Any], budget_B: int) -> int:
    if "validation_size_by_budget" in config:
        value = config["validation_size_by_budget"].get(str(budget_B), config["validation_size_by_budget"].get(budget_B))
        if value is None:
            raise ValueError(f"validation_size_by_budget has no entry for B={budget_B}")
        return int(value)
    return int(round(float(config.get("validation_fraction", 0.2)) * int(budget_B)))


def effective_budget(config: dict[str, Any], budget_B: int) -> int:
    return int(budget_B) - validation_size_for_budget(config, budget_B)


def allocation_grid(config: dict[str, Any]) -> list[float]:
    return [float(x) for x in config["allocation_grid"]]


def s_for_rho(rho: float, n_eff: int) -> int:
    return min(max(1, int(round(float(rho) * int(n_eff)))), int(n_eff) - 1)


def rho_slug(rho: float) -> str:
    return f"rho_{float(rho):.3f}".replace(".", "p")


def allocation_cell_dir(
    config: dict[str, Any],
    method_name: str,
    budget_B: int,
    replication_id: int,
    rho: float,
) -> Path:
    return (
        Path(config["output_dir"])
        / str(method_name).lower()
        / f"B_{int(budget_B):04d}"
        / f"rep_{int(replication_id):02d}"
        / rho_slug(rho)
    )


def build_allocation_split(clean: pd.DataFrame, config: dict[str, Any], budget_B: int, replication_id: int) -> AllocationSplit:
    population = build_population_split(clean, config)
    if split_source(config) != "p_target":
        raise ValueError("full-grid allocation validation requires split_source: p_target")
    source_ids = population.p_target_ids
    if int(budget_B) > len(source_ids):
        raise ValueError(f"B={budget_B} exceeds |P_target|={len(source_ids)}")

    validation_size = validation_size_for_budget(config, budget_B)
    n_eff = int(budget_B) - validation_size
    if validation_size <= 0 or n_eff <= 1:
        raise ValueError(f"invalid validation/effective sizes for B={budget_B}: V={validation_size}, n_eff={n_eff}")
    if max(s_for_rho(rho, n_eff) for rho in allocation_grid(config)) >= n_eff:
        raise ValueError("largest allocation leaves no PPI correction labels")

    rng = np.random.default_rng(allocation_seed(config, replication_id, budget_B))
    labeled_ids = rng.choice(source_ids, size=int(budget_B), replace=False)
    validation_ids = rng.choice(labeled_ids, size=validation_size, replace=False)
    effective_ids = np.setdiff1d(labeled_ids, validation_ids, assume_unique=False)
    train_order_ids = rng.permutation(effective_ids)
    unlabeled_ids = np.setdiff1d(source_ids, labeled_ids, assume_unique=False)
    bundle = AllocationSplit(
        budget_B=int(budget_B),
        validation_ids=np.asarray(validation_ids, dtype=np.int64),
        labeled_ids=np.asarray(labeled_ids, dtype=np.int64),
        effective_ids=np.asarray(effective_ids, dtype=np.int64),
        train_order_ids=np.asarray(train_order_ids, dtype=np.int64),
        unlabeled_ids=np.asarray(unlabeled_ids, dtype=np.int64),
    )
    validate_allocation_split(bundle, config)
    return bundle


def allocation_train_ids_for_rho(bundle: AllocationSplit, rho: float) -> np.ndarray:
    return bundle.train_order_ids[: s_for_rho(rho, len(bundle.effective_ids))]


def allocation_correction_ids_for_rho(bundle: AllocationSplit, rho: float) -> np.ndarray:
    train_ids = set(map(int, allocation_train_ids_for_rho(bundle, rho)))
    return np.asarray([x for x in bundle.effective_ids if int(x) not in train_ids], dtype=np.int64)


def validate_allocation_split(bundle: AllocationSplit, config: dict[str, Any]) -> None:
    labeled = set(map(int, bundle.labeled_ids))
    validation = set(map(int, bundle.validation_ids))
    effective = set(map(int, bundle.effective_ids))
    unlabeled = set(map(int, bundle.unlabeled_ids))
    if len(labeled) != len(bundle.labeled_ids):
        raise ValueError("labeled budget has duplicate sample_ids")
    if len(validation) != len(bundle.validation_ids):
        raise ValueError("validation split has duplicate sample_ids")
    if len(effective) != len(bundle.effective_ids):
        raise ValueError("effective split has duplicate sample_ids")
    if len(unlabeled) != len(bundle.unlabeled_ids):
        raise ValueError("unlabeled split has duplicate sample_ids")
    if not validation.issubset(labeled):
        raise ValueError("validation is not a subset of labeled budget")
    if effective != labeled - validation:
        raise ValueError("effective labels are not exactly labeled minus validation")
    if labeled & unlabeled:
        raise ValueError("unlabeled pool overlaps labeled budget")
    previous: set[int] = set()
    for rho in allocation_grid(config):
        train = set(map(int, allocation_train_ids_for_rho(bundle, rho)))
        correction = set(map(int, allocation_correction_ids_for_rho(bundle, rho)))
        if not train.issubset(effective):
            raise ValueError(f"train set for rho={rho} is not a subset of effective labels")
        if train & correction:
            raise ValueError(f"train/correction overlap for rho={rho}")
        if train | correction != effective:
            raise ValueError(f"train/correction do not cover effective labels for rho={rho}")
        if not previous.issubset(train):
            raise ValueError("allocation train sets are not nested")
        previous = train


def ppi_metrics(
    clean: pd.DataFrame,
    p_target_ids: np.ndarray,
    labeled_ids: np.ndarray,
    correction_pred: pd.DataFrame,
    unlabeled_pred: pd.DataFrame,
) -> dict[str, float | int | bool]:
    correction_residual_raw = correction_pred["residual_raw"].to_numpy(dtype=float)
    correction_residual_scaled = correction_pred["residual_scaled"].to_numpy(dtype=float)
    unlabeled_pred_raw = unlabeled_pred["pred_raw"].to_numpy(dtype=float)
    unlabeled_pred_scaled = unlabeled_pred["pred_scaled"].to_numpy(dtype=float)
    correction_n = int(len(correction_pred))
    unlabeled_n = int(len(unlabeled_pred))

    residual_var_raw = float(np.var(correction_residual_raw, ddof=1))
    residual_var_scaled = float(np.var(correction_residual_scaled, ddof=1))
    pred_var_raw = float(np.var(unlabeled_pred_raw, ddof=1))
    pred_var_scaled = float(np.var(unlabeled_pred_scaled, ddof=1))
    ppi_var_raw = residual_var_raw / correction_n + pred_var_raw / unlabeled_n
    ppi_var_scaled = residual_var_scaled / correction_n + pred_var_scaled / unlabeled_n
    ppi_se_raw = math.sqrt(ppi_var_raw)

    mu_hat_raw = float(np.mean(correction_residual_raw) + np.mean(unlabeled_pred_raw))
    target_points = clean.loc[clean["sample_id"].isin(p_target_ids), Y_COL].astype(float)
    labeled_points = clean.loc[clean["sample_id"].isin(labeled_ids), Y_COL].astype(float)
    target_mean_raw = float(target_points.mean())
    sample_mean_raw = float(labeled_points.mean())
    sample_mean_var_raw = float(labeled_points.var(ddof=1) / len(labeled_points))
    sample_mean_se_raw = math.sqrt(sample_mean_var_raw)

    ci_low = mu_hat_raw - 1.96 * ppi_se_raw
    ci_high = mu_hat_raw + 1.96 * ppi_se_raw
    sample_ci_low = sample_mean_raw - 1.96 * sample_mean_se_raw
    sample_ci_high = sample_mean_raw + 1.96 * sample_mean_se_raw
    return {
        "target_mean_raw": target_mean_raw,
        "mu_hat_ppi_raw": mu_hat_raw,
        "error_raw": mu_hat_raw - target_mean_raw,
        "abs_error_raw": abs(mu_hat_raw - target_mean_raw),
        "correction_size": correction_n,
        "unlabeled_size": unlabeled_n,
        "correction_residual_mean_raw": float(np.mean(correction_residual_raw)),
        "correction_residual_var_raw": residual_var_raw,
        "correction_residual_var_scaled": residual_var_scaled,
        "unlabeled_prediction_mean_raw": float(np.mean(unlabeled_pred_raw)),
        "unlabeled_prediction_var_raw": pred_var_raw,
        "unlabeled_prediction_var_scaled": pred_var_scaled,
        "ppi_var_est_raw": float(ppi_var_raw),
        "ppi_var_est_scaled": float(ppi_var_scaled),
        "ppi_se_raw": float(ppi_se_raw),
        "ppi_ci_low_raw": float(ci_low),
        "ppi_ci_high_raw": float(ci_high),
        "ppi_covered": bool(ci_low <= target_mean_raw <= ci_high),
        "sample_mean_raw": sample_mean_raw,
        "sample_mean_error_raw": sample_mean_raw - target_mean_raw,
        "sample_mean_var_est_raw": sample_mean_var_raw,
        "sample_mean_se_raw": float(sample_mean_se_raw),
        "sample_mean_ci_low_raw": float(sample_ci_low),
        "sample_mean_ci_high_raw": float(sample_ci_high),
        "sample_mean_covered": bool(sample_ci_low <= target_mean_raw <= sample_ci_high),
    }


def write_allocation_manifest(
    output_dir: Path,
    config: dict[str, Any],
    population: Any,
    split: AllocationSplit,
    budget_B: int,
    replication_id: int,
    rho: float,
    method_name: str,
    train_ids: np.ndarray,
    correction_ids: np.ndarray,
) -> None:
    manifest = {
        "budget_B": int(budget_B),
        "replication_id": int(replication_id),
        "rho": float(rho),
        "s_train": int(len(train_ids)),
        "method": str(method_name),
        "split_source": split_source(config),
        "population_counts": {
            "P0": int(len(population.p0_ids)),
            "H_scale": int(len(population.h_scale_ids)),
            "P_target": int(len(population.p_target_ids)),
        },
        "population_hashes": {
            "P0": ids_hash(population.p0_ids),
            "H_scale": ids_hash(population.h_scale_ids),
            "P_target": ids_hash(population.p_target_ids),
        },
        "counts": {
            "N_labeled": int(len(split.labeled_ids)),
            "validation": int(len(split.validation_ids)),
            "N_eff": int(len(split.effective_ids)),
            "train": int(len(train_ids)),
            "correction": int(len(correction_ids)),
            "unlabeled": int(len(split.unlabeled_ids)),
        },
        "hashes": {
            "N_labeled": ids_hash(split.labeled_ids),
            "validation": ids_hash(split.validation_ids),
            "N_eff": ids_hash(split.effective_ids),
            "train": ids_hash(train_ids),
            "correction": ids_hash(correction_ids),
            "unlabeled": ids_hash(split.unlabeled_ids),
        },
    }
    with open(output_dir / "allocation_split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def train_allocation_cell(
    config: dict[str, Any],
    method_name: str,
    budget_B: int,
    replication_id: int,
    rho: float,
) -> Path:
    method = method_by_name(config, method_name)
    output_dir = allocation_cell_dir(config, method.name, budget_B, replication_id, rho)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        print(
            f"skip_completed method={method.name} B={budget_B} rep={replication_id} rho={rho} metrics={metrics_path}",
            flush=True,
        )
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    clean = clean_wine(config["input_csv"])
    population = build_population_split(clean, config)
    split = build_allocation_split(clean, config, budget_B, replication_id)
    train_ids = allocation_train_ids_for_rho(split, rho)
    correction_ids = allocation_correction_ids_for_rho(split, rho)
    train_df = subset_by_ids(clean, train_ids, "train")
    validation_df = subset_by_ids(clean, split.validation_ids, "validation")
    correction_df = subset_by_ids(clean, correction_ids, "ppi_correction")
    unlabeled_df = subset_by_ids(clean, split.unlabeled_ids, "unlabeled")
    write_allocation_manifest(
        output_dir,
        config,
        population,
        split,
        budget_B,
        replication_id,
        rho,
        method.name,
        train_ids,
        correction_ids,
    )

    requested_batch = int(config["batch_size"])
    batch_candidates = [requested_batch]
    fallback_values = config.get("oom_fallback_batch_sizes", [64, 32])
    if not isinstance(fallback_values, list):
        fallback_values = [fallback_values]
    for fallback_value in fallback_values:
        fallback_batch = int(fallback_value)
        if fallback_batch > 0 and fallback_batch not in batch_candidates:
            batch_candidates.append(fallback_batch)

    used_oom_fallback = False
    last_error: str | None = None
    for batch_index, batch_size in enumerate(batch_candidates):
        try:
            print(
                "full_grid_cell_start",
                f"method={method.name}",
                f"loss={method.training_loss}",
                f"stop_metric={method.early_stopping_metric}",
                f"B={budget_B}",
                f"rep={replication_id}",
                f"rho={rho:.3f}",
                f"s={len(train_ids)}",
                f"correction={len(correction_ids)}",
                f"batch_size={batch_size}",
                flush=True,
            )
            runtime, validation_pred, _, extra = train_once(
                config,
                train_df,
                validation_df,
                validation_df,
                batch_size=batch_size,
                seed=allocation_seed(
                    config,
                    replication_id,
                    budget_B,
                    salt=0 if bool(config.get("common_train_seed_within_rep", True)) else len(train_ids),
                ),
                loss_name=method.training_loss,
                early_stopping_metric=method.early_stopping_metric,
                extra_prediction_frames={
                    "correction": correction_df,
                    "unlabeled": unlabeled_df,
                },
            )
            runtime["requested_batch_size"] = requested_batch
            runtime["oom_fallback_used"] = bool(used_oom_fallback)
            break
        except RuntimeError as exc:
            last_error = repr(exc)
            if not is_cuda_oom(exc) or batch_index == len(batch_candidates) - 1:
                raise
            used_oom_fallback = True
            next_batch = batch_candidates[batch_index + 1]
            print(
                f"cuda_oom_retry method={method.name} B={budget_B} rep={replication_id} rho={rho:.3f} next_batch={next_batch}",
                flush=True,
            )
            exc.__traceback__ = None
            del exc
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
    else:
        raise RuntimeError(f"training failed: {last_error}")

    correction_pred = extra["correction"]
    unlabeled_pred = extra["unlabeled"]
    validation_metrics = prediction_metrics(validation_pred)
    ppi = ppi_metrics(clean, population.p_target_ids, split.labeled_ids, correction_pred, unlabeled_pred)
    epoch_history = runtime.pop("epoch_history", [])
    if epoch_history:
        pd.DataFrame(epoch_history).to_csv(output_dir / "epoch_history.csv", index=False)
    if bool(config.get("save_predictions", False)):
        validation_pred.to_parquet(output_dir / "validation_predictions.parquet", index=False)
        correction_pred.to_parquet(output_dir / "correction_predictions.parquet", index=False)
        unlabeled_pred[["sample_id", "pred_scaled", "pred_raw"]].to_parquet(
            output_dir / "unlabeled_predictions.parquet",
            index=False,
        )

    metrics = {
        "method": method.name,
        "training_loss": method.training_loss,
        "early_stopping_metric": method.early_stopping_metric,
        "budget_B": int(budget_B),
        "replication_id": int(replication_id),
        "rho": float(rho),
        "validation_size": int(len(split.validation_ids)),
        "n_eff": int(len(split.effective_ids)),
        "s_train": int(len(train_ids)),
        "correction_size": int(len(correction_ids)),
        "unlabeled_size": int(len(split.unlabeled_ids)),
        "split_source": split_source(config),
        "validation": validation_metrics,
        "ppi": ppi,
        "runtime": runtime,
    }
    write_json(metrics_path, metrics)
    print(
        "full_grid_cell_done",
        f"method={method.name}",
        f"B={budget_B}",
        f"rep={replication_id}",
        f"rho={rho:.3f}",
        f"s={len(train_ids)}",
        f"ppi_var={ppi['ppi_var_est_raw']:.8f}",
        f"metrics={metrics_path}",
        flush=True,
    )
    return output_dir


def all_task_cells(config: dict[str, Any]) -> list[tuple[str, int, int, float]]:
    cells: list[tuple[str, int, int, float]] = []
    for method in configured_methods(config):
        for budget_B in [int(x) for x in config["budgets"]]:
            for replication_id in [int(x) for x in config["replication_ids"]]:
                for rho in allocation_grid(config):
                    cells.append((method.name, budget_B, replication_id, rho))
    return cells


def task_index_to_allocation_cell(config: dict[str, Any], task_index: int) -> tuple[str, int, int, float]:
    cells = all_task_cells(config)
    if int(task_index) < 0 or int(task_index) >= len(cells):
        raise ValueError(f"task_index={task_index} out of range for {len(cells)} allocation cells")
    return cells[int(task_index)]


def flatten_metrics(metrics: dict[str, Any], path: Path) -> dict[str, Any]:
    ppi = metrics["ppi"]
    validation = metrics["validation"]
    runtime = metrics["runtime"]
    row = {
        "path": str(path),
        "method": metrics["method"],
        "training_loss": metrics["training_loss"],
        "early_stopping_metric": metrics["early_stopping_metric"],
        "budget_B": int(metrics["budget_B"]),
        "replication_id": int(metrics["replication_id"]),
        "rho": float(metrics["rho"]),
        "validation_size": int(metrics["validation_size"]),
        "n_eff": int(metrics["n_eff"]),
        "s_train": int(metrics["s_train"]),
        "correction_size": int(metrics["correction_size"]),
        "unlabeled_size": int(metrics["unlabeled_size"]),
        "target_mean_raw": float(ppi["target_mean_raw"]),
        "mu_hat_ppi_raw": float(ppi["mu_hat_ppi_raw"]),
        "error_raw": float(ppi["error_raw"]),
        "abs_error_raw": float(ppi["abs_error_raw"]),
        "correction_residual_var_raw": float(ppi["correction_residual_var_raw"]),
        "unlabeled_prediction_var_raw": float(ppi["unlabeled_prediction_var_raw"]),
        "ppi_var_est_raw": float(ppi["ppi_var_est_raw"]),
        "ppi_se_raw": float(ppi["ppi_se_raw"]),
        "ppi_ci_low_raw": float(ppi["ppi_ci_low_raw"]),
        "ppi_ci_high_raw": float(ppi["ppi_ci_high_raw"]),
        "ppi_covered": bool(ppi["ppi_covered"]),
        "sample_mean_raw": float(ppi["sample_mean_raw"]),
        "sample_mean_error_raw": float(ppi["sample_mean_error_raw"]),
        "sample_mean_var_est_raw": float(ppi["sample_mean_var_est_raw"]),
        "sample_mean_covered": bool(ppi["sample_mean_covered"]),
        "validation_residual_var_raw": float(validation["residual_var_raw"]),
        "validation_rmse_raw": float(validation["rmse_raw"]),
        "validation_corr": float(validation["correlation"]),
        "runtime_seconds": float(runtime["runtime_seconds"]),
        "actual_batch_size": int(runtime["actual_batch_size"]),
        "epochs_trained": int(runtime["epochs_trained"]),
        "best_epoch": int(runtime["best_epoch"]),
        "early_stopped": bool(runtime["early_stopped"]),
        "oom_fallback_used": bool(runtime.get("oom_fallback_used", False)),
        "device": runtime.get("device", ""),
    }
    return row


def load_allocation_metrics(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    missing = []
    for method_name, budget_B, replication_id, rho in all_task_cells(config):
        path = allocation_cell_dir(config, method_name, budget_B, replication_id, rho) / "metrics.json"
        if not path.exists():
            missing.append((method_name, budget_B, replication_id, rho, str(path)))
            continue
        with open(path, "r", encoding="utf-8") as f:
            rows.append(flatten_metrics(json.load(f), path))
    if missing:
        preview = "\n".join(map(str, missing[:20]))
        raise FileNotFoundError(f"missing {len(missing)} allocation cells; first missing:\n{preview}")
    return pd.DataFrame(rows).sort_values(["method", "budget_B", "replication_id", "rho"]).reset_index(drop=True)


def scaling_params_raw(config: dict[str, Any]) -> dict[str, float]:
    params = config["scaling_law_params_raw"]
    return {"a": float(params["a"]), "alpha": float(params["alpha"]), "b": float(params["b"])}


def theoretical_allocation(config: dict[str, Any], budget_B: int) -> dict[str, float | int]:
    params = scaling_params_raw(config)
    n_eff = effective_budget(config, budget_B)
    candidates = np.arange(1, n_eff, dtype=int)
    objective = (params["a"] * candidates.astype(float) ** (-params["alpha"]) + params["b"]) / (n_eff - candidates)
    idx = int(np.argmin(objective))
    s_star = int(candidates[idx])
    rho_star = float(s_star / n_eff)
    grid = allocation_grid(config)
    grid_s = np.asarray([s_for_rho(rho, n_eff) for rho in grid], dtype=int)
    nearest_idx = int(np.argmin(np.abs(grid_s - s_star)))
    return {
        "theory_s": s_star,
        "theory_rho": rho_star,
        "theory_objective_raw": float(objective[idx]),
        "theory_eval_rho": float(grid[nearest_idx]),
        "theory_eval_s": int(grid_s[nearest_idx]),
    }


def aggregate_allocation(config: dict[str, Any]) -> dict[str, Path]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_allocation_metrics(config)
    metrics.to_csv(output_dir / "full_grid_cell_metrics.csv", index=False)

    by_rho = (
        metrics.groupby(["budget_B", "method", "rho", "s_train"], as_index=False)
        .agg(
            n_replications=("replication_id", "nunique"),
            mean_ppi_var_est_raw=("ppi_var_est_raw", "mean"),
            se_ppi_var_est_raw=("ppi_var_est_raw", lambda x: float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else math.nan),
            mean_mu_hat_ppi_raw=("mu_hat_ppi_raw", "mean"),
            mean_target_mean_raw=("target_mean_raw", "mean"),
            mean_sample_mean_raw=("sample_mean_raw", "mean"),
            sd_mu_hat_ppi_raw=("mu_hat_ppi_raw", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else math.nan),
            mean_error_raw=("error_raw", "mean"),
            rmse_error_raw=("error_raw", lambda x: float(math.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))),
            coverage=("ppi_covered", "mean"),
            mean_sample_var_est_raw=("sample_mean_var_est_raw", "mean"),
            sample_mean_coverage=("sample_mean_covered", "mean"),
        )
        .sort_values(["budget_B", "method", "rho"])
    )
    by_rho["ci95_low_mu_hat_ppi_raw"] = by_rho["mean_mu_hat_ppi_raw"] - 1.96 * by_rho["sd_mu_hat_ppi_raw"] / np.sqrt(by_rho["n_replications"])
    by_rho["ci95_high_mu_hat_ppi_raw"] = by_rho["mean_mu_hat_ppi_raw"] + 1.96 * by_rho["sd_mu_hat_ppi_raw"] / np.sqrt(by_rho["n_replications"])
    by_rho["var_ratio_to_sample_mean"] = by_rho["mean_ppi_var_est_raw"] / by_rho["mean_sample_var_est_raw"]
    by_rho.to_csv(output_dir / "full_grid_by_rho_summary.csv", index=False)

    rows = []
    for budget_B in [int(x) for x in config["budgets"]]:
        theory = theoretical_allocation(config, budget_B)
        budget_rows = by_rho[by_rho["budget_B"] == budget_B]
        for method_name in sorted(budget_rows["method"].unique()):
            method_rows = budget_rows[budget_rows["method"] == method_name].copy()
            grid_best_idx = method_rows["mean_ppi_var_est_raw"].astype(float).idxmin()
            grid_best = method_rows.loc[grid_best_idx]
            if method_name == "var_stop_var":
                eval_rho = float(theory["theory_eval_rho"])
            else:
                eval_rho = float(grid_best["rho"])
            eval_rows = method_rows[np.isclose(method_rows["rho"].astype(float), eval_rho)]
            if eval_rows.empty:
                raise RuntimeError(f"no eval row for B={budget_B}, method={method_name}, rho={eval_rho}")
            eval_row = eval_rows.iloc[0]
            regret = (
                (float(eval_row["mean_ppi_var_est_raw"]) - float(grid_best["mean_ppi_var_est_raw"]))
                / float(grid_best["mean_ppi_var_est_raw"])
                if method_name == "var_stop_var"
                else math.nan
            )
            rows.append(
                {
                    "budget_B": budget_B,
                    "validation_size": validation_size_for_budget(config, budget_B),
                    "n_eff": effective_budget(config, budget_B),
                    "method": method_name,
                    "theory_rho": float(theory["theory_rho"]) if method_name == "var_stop_var" else math.nan,
                    "theory_s": int(theory["theory_s"]) if method_name == "var_stop_var" else math.nan,
                    "grid_best_rho": float(grid_best["rho"]),
                    "grid_best_s": int(grid_best["s_train"]),
                    "eval_rho": eval_rho,
                    "eval_s": int(eval_row["s_train"]),
                    "var_ratio": float(eval_row["var_ratio_to_sample_mean"]),
                    "regret": float(regret) if np.isfinite(regret) else math.nan,
                    "coverage": float(eval_row["coverage"]),
                    "mean_error_raw": float(eval_row["mean_error_raw"]),
                    "rmse_error_raw": float(eval_row["rmse_error_raw"]),
                    "mean_ppi_var_est_raw": float(eval_row["mean_ppi_var_est_raw"]),
                    "sample_mean_var_est_raw": float(eval_row["mean_sample_var_est_raw"]),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "full_grid_allocation_summary.csv", index=False)
    write_allocation_figures(config, by_rho, summary, output_dir)
    return {
        "cell_metrics": output_dir / "full_grid_cell_metrics.csv",
        "by_rho_summary": output_dir / "full_grid_by_rho_summary.csv",
        "allocation_summary": output_dir / "full_grid_allocation_summary.csv",
    }


def write_allocation_figures(config: dict[str, Any], by_rho: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = {"var_stop_var": "#1B4E7A", "mse_stop_mse": "#B33A2E"}
    labels = {"var_stop_var": "FT+PPI (Var Loss)", "mse_stop_mse": "FT+PPI (MSE Loss)"}
    for budget_B in [int(x) for x in config["budgets"]]:
        budget_rows = by_rho[by_rho["budget_B"] == budget_B]
        if budget_rows.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
        for ax, method_name in zip(axes, ["var_stop_var", "mse_stop_mse"]):
            method_rows = budget_rows[budget_rows["method"] == method_name].sort_values("rho")
            if method_rows.empty:
                ax.axis("off")
                continue
            x = method_rows["rho"].to_numpy(dtype=float) * 100.0
            y = method_rows["mean_mu_hat_ppi_raw"].to_numpy(dtype=float)
            low = method_rows["ci95_low_mu_hat_ppi_raw"].to_numpy(dtype=float)
            high = method_rows["ci95_high_mu_hat_ppi_raw"].to_numpy(dtype=float)
            color = colors.get(method_name, "#333333")
            ax.fill_between(x, low, high, color=color, alpha=0.16, label=f"{labels.get(method_name, method_name)} 95% CI")
            ax.plot(x, y, marker="o", color=color, linewidth=2.0, markersize=4.8, label=labels.get(method_name, method_name))
            sample_var = float(method_rows["mean_sample_var_est_raw"].iloc[0])
            sample_center = float(method_rows["mean_sample_mean_raw"].iloc[0])
            target_center = float(method_rows["mean_target_mean_raw"].iloc[0])
            sample_hw = 1.96 * math.sqrt(sample_var)
            ax.axhspan(sample_center - sample_hw, sample_center + sample_hw, color="#8A8F93", alpha=0.12, label="Sample Mean 95% CI")
            ax.axhline(target_center, color="#707070", linewidth=1.2, label="Ground Truth")
            if method_name == "var_stop_var":
                theory = theoretical_allocation(config, budget_B)
                ax.scatter(
                    [float(theory["theory_rho"]) * 100.0],
                    [np.interp(float(theory["theory_rho"]), method_rows["rho"], y)],
                    marker="*",
                    s=260,
                    color="#FFD92F",
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=5,
                    label=f"Theoretical Optimal (Ratio ~= {float(theory['theory_rho']):.1%})",
                )
            ax.set_xlabel("Fine-tuning Allocation (%)", fontweight="bold")
            ax.set_ylabel("Estimated Mean", fontweight="bold")
            ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        axes[0].set_title("(a) Variance-loss-based fine-tuning", y=-0.30)
        axes[1].set_title("(b) MSE-loss-based fine-tuning", y=-0.30)
        fig.suptitle(f"Performance of FT+PPI estimators under different sample allocations, B={budget_B}", y=0.99)
        fig.tight_layout(rect=[0, 0.08, 1, 0.96])
        fig.savefig(figures_dir / f"full_grid_allocation_B{budget_B}_estimated_mean.png", dpi=240)
        fig.savefig(figures_dir / f"full_grid_allocation_B{budget_B}_estimated_mean.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        for method_name, method_rows in budget_rows.groupby("method"):
            method_rows = method_rows.sort_values("rho")
            ax.plot(
                method_rows["rho"].to_numpy(dtype=float) * 100.0,
                method_rows["var_ratio_to_sample_mean"].to_numpy(dtype=float),
                marker="o",
                linewidth=2.0,
                color=colors.get(method_name, "#333333"),
                label=labels.get(method_name, method_name),
            )
        theory = theoretical_allocation(config, budget_B)
        ax.axvline(float(theory["theory_rho"]) * 100.0, color="#E66100", linestyle="--", linewidth=1.2, label="LoRA-Var theoretical")
        ax.axhline(1.0, color="#777777", linestyle=":", linewidth=1.0, label="Sample Mean")
        ax.set_xlabel("Fine-tuning Allocation (%)", fontweight="bold")
        ax.set_ylabel("Normalized Estimated Variance", fontweight="bold")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / f"full_grid_allocation_B{budget_B}_variance_objective.png", dpi=240)
        fig.savefig(figures_dir / f"full_grid_allocation_B{budget_B}_variance_objective.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wine Reviews full-grid allocation validation.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train-cell")
    train_parser.add_argument("--config", required=True)
    train_group = train_parser.add_mutually_exclusive_group(required=True)
    train_group.add_argument("--task-index", type=int)
    train_group.add_argument("--cell", nargs=4, metavar=("METHOD", "B", "REP", "RHO"))

    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--config", required=True)

    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "train-cell":
        if args.task_index is not None:
            method_name, budget_B, replication_id, rho = task_index_to_allocation_cell(config, args.task_index)
        else:
            method_name = str(args.cell[0])
            budget_B = int(args.cell[1])
            replication_id = int(args.cell[2])
            rho = float(args.cell[3])
        train_allocation_cell(config, method_name, budget_B, replication_id, rho)
    elif args.command == "aggregate":
        outputs = aggregate_allocation(config)
        print("full_grid_allocation_aggregate_done")
        for name, path in outputs.items():
            print(f"{name}={path}")


if __name__ == "__main__":
    main()
