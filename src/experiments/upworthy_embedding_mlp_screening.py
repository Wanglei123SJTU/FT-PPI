from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.formatting import dataframe_to_markdown


Y_COL = "y_logit_ctr_diff"
DEFAULT_TARGETS = [
    "delta_vader_compound",
    "delta_vader_neg",
    "delta_curiosity_share",
    "delta_log_word_count",
]
DEFAULT_CONTROLS = [
    "delta_question",
    "delta_numeric",
    "delta_log_word_count",
    "delta_vader_compound",
    "delta_curiosity_share",
    "delta_common_zipf",
    "delta_context_coverage",
]
DEFAULT_S_GRID = [0, 200, 500, 1000, 3000]
DEFAULT_BUDGETS = [500, 1000, 1500, 3000, 5000]


@dataclass(frozen=True)
class MethodSpec:
    method: str
    objective: str
    stop_metric: str


DEFAULT_METHODS = [
    MethodSpec("mse_stop_mse", "mse", "mse"),
    MethodSpec("mse_stop_ifvar", "mse", "ifvar"),
    MethodSpec("ifvar_stop_ifvar", "ifvar", "ifvar"),
]


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_methods(values: Iterable[str]) -> list[MethodSpec]:
    by_name = {method.method: method for method in DEFAULT_METHODS}
    methods: list[MethodSpec] = []
    for value in values:
        if value in by_name:
            methods.append(by_name[value])
            continue
        pieces = value.split(":")
        if len(pieces) != 3:
            raise ValueError(f"custom method must be name:objective:stop_metric, got {value}")
        methods.append(MethodSpec(pieces[0], pieces[1], pieces[2]))
    return methods


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = {"embedding_a", "embedding_b", "screen_row_id", "screen_split", "y"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
    return {key: data[key] for key in data.files}


def pair_embeddings(cache: dict[str, np.ndarray], representation: str) -> np.ndarray:
    a = cache["embedding_a"].astype(np.float32)
    b = cache["embedding_b"].astype(np.float32)
    if a.shape != b.shape:
        raise ValueError(f"embedding shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    if representation == "diff":
        return diff.astype(np.float32)
    if representation == "diff_abs":
        return np.concatenate([diff, np.abs(diff)], axis=1).astype(np.float32)
    raise ValueError(f"unknown representation: {representation}")


def standardize_embeddings(
    embeddings: np.ndarray,
    train_pool_indices: np.ndarray,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = embeddings[train_pool_indices].mean(axis=0)
    sd = embeddings[train_pool_indices].std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return ((embeddings - mean) / sd).astype(np.float32), mean.astype(np.float32), sd.astype(np.float32)


def design_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(len(frame)), frame[columns].astype(float).to_numpy()])


def fit_eval_ols(
    frame: pd.DataFrame,
    *,
    target: str,
    controls: list[str],
    y_col: str = Y_COL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    columns = [target, *controls]
    x = design_matrix(frame, columns)
    y = frame[y_col].astype(float).to_numpy()
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    residual = y - x @ beta
    hessian = x.T @ x / len(frame)
    return beta, hessian, residual, columns


def if_weights_for_frame(frame: pd.DataFrame, *, columns: list[str], hessian: np.ndarray, target: str, ridge: float) -> np.ndarray:
    x = design_matrix(frame, columns)
    target_idx = columns.index(target) + 1
    h = hessian.copy()
    if ridge > 0:
        h = h + ridge * np.eye(h.shape[0])
    try:
        h_inv = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        h_inv = np.linalg.pinv(h)
    return x @ h_inv[target_idx, :]


def eligible_controls(
    frame: pd.DataFrame,
    metadata: pd.DataFrame | None,
    *,
    target: str,
    controls: list[str],
    corr_threshold: float = 0.85,
) -> list[str]:
    meta_family: dict[str, str] = {}
    if metadata is not None and {"feature", "family"}.issubset(metadata.columns):
        meta_family = dict(zip(metadata["feature"], metadata["family"]))
    target_family = meta_family.get(target)
    selected: list[str] = []
    for control in controls:
        if control == target or control not in frame.columns:
            continue
        if target_family is not None and meta_family.get(control) == target_family:
            continue
        corr = frame[[target, control]].corr().iloc[0, 1]
        if np.isfinite(corr) and abs(float(corr)) >= corr_threshold:
            continue
        selected.append(control)
    return selected


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, if_weights: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    if_resid = if_weights * residual
    return {
        "mse": float(np.mean(residual**2)),
        "ifvar": float(np.var(if_resid, ddof=0)),
        "ifmean": float(np.mean(if_resid)),
    }


def _build_torch_model(input_dim: int, hidden_dim: int, dropout: float):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_dim, 1),
    )


def _torch_predict(model, x, *, antisymmetric: bool):
    pred = model(x).reshape(-1)
    if not antisymmetric:
        return pred
    pred_neg = model(-x).reshape(-1)
    return 0.5 * (pred - pred_neg)


def train_one_mlp(
    embeddings: np.ndarray,
    y_scaled: np.ndarray,
    if_weights: np.ndarray,
    *,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    method: MethodSpec,
    seed: int,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: str,
    antisymmetric: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.manual_seed_all(seed)
    selected_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = _build_torch_model(embeddings.shape[1], hidden_dim, dropout).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    x_all = torch.from_numpy(embeddings.astype(np.float32)).to(selected_device)
    y_all = torch.from_numpy(y_scaled.astype(np.float32)).to(selected_device)
    a_all = torch.from_numpy(if_weights.astype(np.float32)).to(selected_device)
    train_tensor = torch.from_numpy(train_indices.astype(np.int64)).to(selected_device)
    validation_tensor = torch.from_numpy(validation_indices.astype(np.int64)).to(selected_device)

    best_state = None
    best_metric = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    batch_size = max(1, int(batch_size))

    for epoch in range(max_epochs):
        model.train()
        perm = train_tensor[torch.randperm(len(train_tensor), device=selected_device)]
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            pred = _torch_predict(model, x_all[idx], antisymmetric=antisymmetric)
            residual = y_all[idx] - pred
            if method.objective == "mse":
                loss = torch.mean(residual**2)
            elif method.objective == "ifvar":
                if_resid = a_all[idx] * residual
                loss = torch.mean((if_resid - torch.mean(if_resid)) ** 2)
            else:
                raise ValueError(f"unknown objective: {method.objective}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            val_pred = _torch_predict(model, x_all[validation_tensor], antisymmetric=antisymmetric)
            val_residual = y_all[validation_tensor] - val_pred
            val_mse = torch.mean(val_residual**2).item()
            val_if = a_all[validation_tensor] * val_residual
            val_ifvar = torch.var(val_if, unbiased=False).item()
            metric = val_mse if method.stop_metric == "mse" else val_ifvar

        if metric < best_metric - 1e-7:
            best_metric = metric
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        pred_all = _torch_predict(model, x_all, antisymmetric=antisymmetric).detach().cpu().numpy()
    diagnostics = {
        "best_epoch": float(best_epoch),
        "best_validation_metric": float(best_metric),
        "device": str(selected_device),
    }
    return pred_all.astype(np.float32), diagnostics


def select_train_validation(
    train_pool_indices: np.ndarray,
    *,
    s: int,
    seed: int,
    min_validation: int,
) -> tuple[np.ndarray, np.ndarray]:
    if s <= 0:
        raise ValueError("s must be positive for train/validation selection")
    if s + min_validation > len(train_pool_indices):
        raise ValueError(
            f"s={s} plus min_validation={min_validation} exceeds train pool size {len(train_pool_indices)}"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_pool_indices)
    train = np.sort(perm[:s])
    validation = np.sort(perm[s:])
    return train, validation


def budget_rows_for_prediction(
    *,
    target: str,
    method: str,
    s: int,
    ifvar: float,
    direct_ols_ifvar: float,
    zero_ifvar: float,
    budgets: list[int],
) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    for budget in budgets:
        if s >= budget:
            continue
        direct_ratio = (ifvar / (budget - s)) / (direct_ols_ifvar / budget) if direct_ols_ifvar > 0 else np.nan
        zero_ratio = (ifvar / (budget - s)) / (zero_ifvar / budget) if zero_ifvar > 0 else np.nan
        rows.append(
            {
                "target": target,
                "method": method,
                "s": int(s),
                "budget": int(budget),
                "ratio_vs_direct_ols": float(direct_ratio),
                "ratio_vs_zero_surrogate": float(zero_ratio),
                "beats_direct_ols": bool(np.isfinite(direct_ratio) and direct_ratio < 1.0),
                "beats_zero_surrogate": bool(np.isfinite(zero_ratio) and zero_ratio < 1.0),
            }
        )
    return rows


def run_screening(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = load_npz(args.embedding_npz)
    frame = pd.read_csv(args.sample_csv)
    metadata = pd.read_csv(args.feature_metadata_csv) if args.feature_metadata_csv and Path(args.feature_metadata_csv).exists() else None
    if "screen_split" not in frame.columns:
        frame["screen_split"] = cache["screen_split"].astype(str)
    if len(frame) != len(cache["y"]):
        raise ValueError(f"sample CSV rows {len(frame)} do not match embedding rows {len(cache['y'])}")

    targets = parse_csv_list(args.targets)
    controls = parse_csv_list(args.controls)
    s_grid = parse_int_list(args.s_grid)
    budgets = parse_int_list(args.budgets)
    methods = parse_methods(args.methods)
    embeddings_raw = pair_embeddings(cache, args.representation)
    train_pool_indices = np.flatnonzero(frame["screen_split"].astype(str).to_numpy() == "train_pool")
    evaluation_indices = np.flatnonzero(frame["screen_split"].astype(str).to_numpy() == "evaluation")
    if len(train_pool_indices) == 0 or len(evaluation_indices) == 0:
        raise ValueError("need non-empty train_pool and evaluation screen splits")
    embeddings, embed_mean, embed_sd = standardize_embeddings(embeddings_raw, train_pool_indices)
    y = frame[Y_COL].astype(float).to_numpy()
    y_mean = float(y[train_pool_indices].mean()) if args.standardize_y else 0.0
    y_sd = float(y[train_pool_indices].std(ddof=0)) if args.standardize_y else 1.0
    if not np.isfinite(y_sd) or y_sd <= 1e-8:
        y_sd = 1.0
    y_scaled = ((y - y_mean) / y_sd).astype(np.float32)

    result_rows: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    eval_frame = frame.iloc[evaluation_indices].copy()

    for target in targets:
        if target not in frame.columns:
            raise ValueError(f"target feature {target} not found in sample CSV")
        selected_controls = eligible_controls(eval_frame, metadata, target=target, controls=controls)
        beta, hessian, ols_residual, x_columns = fit_eval_ols(eval_frame, target=target, controls=selected_controls)
        weights_all = if_weights_for_frame(frame, columns=x_columns, hessian=hessian, target=target, ridge=args.hessian_ridge)
        weights_eval = weights_all[evaluation_indices]
        y_eval = y[evaluation_indices]
        zero_pred_eval = np.zeros_like(y_eval)
        zero_metrics = evaluate_predictions(y_eval, zero_pred_eval, weights_eval)
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
                "n_train_pool": int(len(train_pool_indices)),
                "n_evaluation": int(len(evaluation_indices)),
            }
        )
        result_rows.append(
            {
                "target": target,
                "method": "zero_surrogate",
                "objective": "none",
                "stop_metric": "none",
                "replication": -1,
                "s": 0,
                "mse": zero_metrics["mse"],
                "ifvar": zero_metrics["ifvar"],
                "ifmean": zero_metrics["ifmean"],
                "ifvar_ratio_vs_zero": 1.0,
                "ifvar_ratio_vs_direct_ols": zero_metrics["ifvar"] / direct_ols_ifvar if direct_ols_ifvar > 0 else np.nan,
                "best_epoch": np.nan,
                "best_validation_metric": np.nan,
                "seconds": 0.0,
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
                print(f"skip target={target} s={s}: not enough train_pool rows", flush=True)
                continue
            for rep in range(args.replications):
                train_indices, validation_indices = select_train_validation(
                    train_pool_indices,
                    s=s,
                    seed=args.seed + 1009 * rep + 17 * s,
                    min_validation=args.min_validation,
                )
                for method in methods:
                    start = time.time()
                    pred_scaled, diagnostics = train_one_mlp(
                        embeddings,
                        y_scaled,
                        weights_all,
                        train_indices=train_indices,
                        validation_indices=validation_indices,
                        method=method,
                        seed=args.seed + 10_000 * rep + s,
                        hidden_dim=args.hidden_dim,
                        dropout=args.dropout,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        batch_size=args.batch_size,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        device=args.device,
                        antisymmetric=args.antisymmetric,
                    )
                    pred = pred_scaled * y_sd + y_mean
                    metrics = evaluate_predictions(y_eval, pred[evaluation_indices], weights_eval)
                    elapsed = time.time() - start
                    result_rows.append(
                        {
                            "target": target,
                            "method": method.method,
                            "objective": method.objective,
                            "stop_metric": method.stop_metric,
                            "replication": rep,
                            "s": int(s),
                            "mse": metrics["mse"],
                            "ifvar": metrics["ifvar"],
                            "ifmean": metrics["ifmean"],
                            "ifvar_ratio_vs_zero": metrics["ifvar"] / zero_metrics["ifvar"] if zero_metrics["ifvar"] > 0 else np.nan,
                            "ifvar_ratio_vs_direct_ols": metrics["ifvar"] / direct_ols_ifvar if direct_ols_ifvar > 0 else np.nan,
                            "best_epoch": diagnostics["best_epoch"],
                            "best_validation_metric": diagnostics["best_validation_metric"],
                            "seconds": elapsed,
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
                        f"ifvar_ratio_vs_zero={metrics['ifvar'] / zero_metrics['ifvar']:.4f}",
                        flush=True,
                    )
    return pd.DataFrame(result_rows), pd.DataFrame(budget_rows), pd.DataFrame(target_rows)


def write_report(
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
            mean_seconds=("seconds", "mean"),
        )
    )
    zero = results.loc[results["method"] == "zero_surrogate", ["target", "ifvar", "mse"]].rename(
        columns={"ifvar": "zero_ifvar", "mse": "zero_mse"}
    )
    summary = summary.merge(zero, on="target", how="left")
    best = (
        summary.sort_values(["target", "mean_ifvar_ratio_vs_zero"])
        .groupby("target", as_index=False)
        .head(3)
        .copy()
    )
    budget_summary = (
        budgets.groupby(["target", "method", "s", "budget"], as_index=False)
        .agg(
            mean_ratio_vs_direct_ols=("ratio_vs_direct_ols", "mean"),
            mean_ratio_vs_zero_surrogate=("ratio_vs_zero_surrogate", "mean"),
            win_rate_direct_ols=("beats_direct_ols", "mean"),
            win_rate_zero_surrogate=("beats_zero_surrogate", "mean"),
        )
        .sort_values(["target", "budget", "mean_ratio_vs_direct_ols"])
    )
    results.to_csv(output_dir / "embedding_mlp_results.csv", index=False)
    summary.to_csv(output_dir / "embedding_mlp_summary.csv", index=False)
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
            "methods": [method.__dict__ for method in parse_methods(args.methods)],
            "replications": int(args.replications),
            "hidden_dim": int(args.hidden_dim),
            "batch_size": int(args.batch_size),
            "antisymmetric": bool(args.antisymmetric),
            "standardize_y": bool(args.standardize_y),
        },
    )
    with open(output_dir / "screening_report.md", "w", encoding="utf-8") as f:
        f.write("# Upworthy OpenAI Embedding + MLP Screening\n\n")
        f.write("This is a semantic-surrogate feasibility screen, not final LoRA evidence.\n\n")
        f.write("## Target Diagnostics\n\n")
        f.write(dataframe_to_markdown(targets.round(4)))
        f.write("\n\n## Best Variance Reductions\n\n")
        f.write(dataframe_to_markdown(best.round(4)))
        f.write("\n\n## Budget Win Summary: Best Rows\n\n")
        f.write(dataframe_to_markdown(budget_summary.groupby(["target", "budget"], as_index=False).head(1).round(4)))
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen Upworthy coefficient targets with cached OpenAI embeddings and a small MLP.")
    parser.add_argument("--embedding-npz", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--feature-metadata-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/upworthy_m_estimation/openai_embedding_mlp_screening"))
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--controls", default=",".join(DEFAULT_CONTROLS))
    parser.add_argument("--s-grid", default=",".join(str(item) for item in DEFAULT_S_GRID))
    parser.add_argument("--budgets", default=",".join(str(item) for item in DEFAULT_BUDGETS))
    parser.add_argument("--methods", nargs="+", default=[method.method for method in DEFAULT_METHODS])
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--min-validation", type=int, default=500)
    parser.add_argument("--representation", choices=["diff", "diff_abs"], default="diff")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hessian-ridge", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--no-antisymmetric", dest="antisymmetric", action="store_false")
    parser.add_argument("--no-standardize-y", dest="standardize_y", action="store_false")
    parser.set_defaults(antisymmetric=True, standardize_y=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, budgets, targets = run_screening(args)
    write_report(args.output_dir, results, budgets, targets, args)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
