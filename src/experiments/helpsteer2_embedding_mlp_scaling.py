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

from src.experiments.helpsteer2_preference_regression import Y_COL


DEFAULT_FEATURES = ["delta_log_length", "delta_prompt_coverage", "delta_format"]
DEFAULT_TARGET = "delta_format"
DEFAULT_S_GRID = [0, 50, 100, 250, 500, 750, 1000, 1500, 3000]
DEFAULT_BUDGETS = [500, 1000, 1500, 3000]


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


def load_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "pair_embedding" not in data:
        raise ValueError(f"missing pair_embedding in {path}")
    return {key: data[key] for key in data.files}


def compute_if_weights(
    frame: pd.DataFrame,
    *,
    target: str,
    feature_columns: list[str],
    ridge: float = 0.0,
) -> np.ndarray:
    if target not in feature_columns:
        raise ValueError(f"target={target} must be in feature_columns={feature_columns}")
    design = np.column_stack([np.ones(len(frame)), frame[feature_columns].astype(float).to_numpy()])
    hessian = design.T @ design / len(frame)
    if ridge > 0:
        hessian = hessian + ridge * np.eye(hessian.shape[0])
    try:
        hessian_inv = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        hessian_inv = np.linalg.pinv(hessian)
    target_idx = feature_columns.index(target) + 1
    return design @ hessian_inv[target_idx, :]


def split_indices(
    n: int,
    *,
    seed: int,
    train_frac: float = 0.60,
    validation_frac: float = 0.20,
) -> dict[str, np.ndarray]:
    if train_frac <= 0 or validation_frac <= 0 or train_frac + validation_frac >= 1:
        raise ValueError("train_frac and validation_frac must be positive and sum to less than 1")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(train_frac * n))
    n_val = int(round(validation_frac * n))
    return {
        "train_pool": np.sort(perm[:n_train]),
        "validation": np.sort(perm[n_train : n_train + n_val]),
        "evaluation": np.sort(perm[n_train + n_val :]),
    }


def standardize_embeddings(
    embeddings: np.ndarray,
    train_indices: np.ndarray,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = embeddings[train_indices].mean(axis=0)
    sd = embeddings[train_indices].std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return ((embeddings - mean) / sd).astype(np.float32), mean.astype(np.float32), sd.astype(np.float32)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, if_weights: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    if_residual = if_weights * residual
    return {
        "mse": float(np.mean(residual**2)),
        "ifvar": float(np.var(if_residual, ddof=0)),
        "ifmean": float(np.mean(if_residual)),
    }


def antisymmetric_from_scores(score_h: np.ndarray, score_neg_h: np.ndarray) -> np.ndarray:
    if score_h.shape != score_neg_h.shape:
        raise ValueError(f"score shape mismatch: {score_h.shape} vs {score_neg_h.shape}")
    return 0.5 * (score_h - score_neg_h)


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


def train_one_model(
    embeddings: np.ndarray,
    y: np.ndarray,
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
    y_all = torch.from_numpy(y.astype(np.float32)).to(selected_device)
    a_all = torch.from_numpy(if_weights.astype(np.float32)).to(selected_device)
    train_tensor = torch.from_numpy(train_indices.astype(np.int64)).to(selected_device)
    validation_tensor = torch.from_numpy(validation_indices.astype(np.int64)).to(selected_device)

    best_state = None
    best_metric = math.inf
    best_epoch = -1
    epochs_without_improvement = 0

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
        "epochs_run": float(epoch + 1),
    }
    return pred_all.astype(np.float32), diagnostics


def run_scaling_experiment(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    target: str = DEFAULT_TARGET,
    feature_columns: list[str] = DEFAULT_FEATURES,
    s_grid: list[int] = DEFAULT_S_GRID,
    replications: int = 10,
    seed: int = 20260621,
    methods: list[MethodSpec] = DEFAULT_METHODS,
    hidden_dim: int = 128,
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    max_epochs: int = 300,
    patience: int = 30,
    device: str = "cuda",
    hessian_ridge: float = 0.0,
    antisymmetric: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(frame) != embeddings.shape[0]:
        raise ValueError(f"frame rows {len(frame)} != embeddings rows {embeddings.shape[0]}")
    y = frame[Y_COL].astype(float).to_numpy()
    if_weights = compute_if_weights(frame, target=target, feature_columns=feature_columns, ridge=hessian_ridge)
    splits = split_indices(len(frame), seed=seed)
    embeddings_std, _, _ = standardize_embeddings(embeddings, splits["train_pool"])
    eval_idx = splits["evaluation"]
    validation_idx = splits["validation"]
    train_pool = splits["train_pool"]

    baseline_pred = np.zeros(len(frame), dtype=np.float32)
    baseline_eval = evaluate_predictions(y[eval_idx], baseline_pred[eval_idx], if_weights[eval_idx])
    rows: list[dict] = []
    rows.append(
        {
            "method": "zero_surrogate",
            "objective": "none",
            "stop_metric": "none",
            "s": 0,
            "replication": -1,
            "eval_mse": baseline_eval["mse"],
            "eval_ifvar": baseline_eval["ifvar"],
            "eval_ifmean": baseline_eval["ifmean"],
            "val_mse": evaluate_predictions(y[validation_idx], baseline_pred[validation_idx], if_weights[validation_idx])[
                "mse"
            ],
            "val_ifvar": evaluate_predictions(
                y[validation_idx], baseline_pred[validation_idx], if_weights[validation_idx]
            )["ifvar"],
            "best_epoch": 0,
            "epochs_run": 0,
            "seconds": 0.0,
        }
    )

    rng = np.random.default_rng(seed)
    for rep in range(replications):
        rep_seed = int(rng.integers(1, 2**31 - 1))
        rep_rng = np.random.default_rng(rep_seed)
        for s in s_grid:
            if s == 0:
                continue
            if s > len(train_pool):
                continue
            train_idx = np.sort(rep_rng.choice(train_pool, size=s, replace=False))
            for method in methods:
                start = time.time()
                pred, diagnostics = train_one_model(
                    embeddings_std,
                    y,
                    if_weights,
                    train_indices=train_idx,
                    validation_indices=validation_idx,
                    method=method,
                    seed=rep_seed + s + len(method.method),
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                    device=device,
                    antisymmetric=antisymmetric,
                )
                eval_metrics = evaluate_predictions(y[eval_idx], pred[eval_idx], if_weights[eval_idx])
                val_metrics = evaluate_predictions(y[validation_idx], pred[validation_idx], if_weights[validation_idx])
                rows.append(
                    {
                        "method": method.method,
                        "objective": method.objective,
                        "stop_metric": method.stop_metric,
                        "s": s,
                        "replication": rep,
                        "eval_mse": eval_metrics["mse"],
                        "eval_ifvar": eval_metrics["ifvar"],
                        "eval_ifmean": eval_metrics["ifmean"],
                        "val_mse": val_metrics["mse"],
                        "val_ifvar": val_metrics["ifvar"],
                        "best_epoch": diagnostics["best_epoch"],
                        "epochs_run": diagnostics["epochs_run"],
                        "seconds": time.time() - start,
                        "antisymmetric": antisymmetric,
                    }
                )
                print(
                    {
                        "method": method.method,
                        "s": s,
                        "replication": rep,
                        "eval_ifvar": eval_metrics["ifvar"],
                        "eval_mse": eval_metrics["mse"],
                    },
                    flush=True,
                )

    result = pd.DataFrame(rows)
    baseline_ifvar = float(baseline_eval["ifvar"])
    baseline_mse = float(baseline_eval["mse"])
    summary_rows: list[dict] = []
    for (method, s), group in result.groupby(["method", "s"], sort=True):
        summary_rows.append(
            {
                "method": method,
                "s": int(s),
                "replications": int(len(group)),
                "eval_ifvar_mean": float(group["eval_ifvar"].mean()),
                "eval_ifvar_sd": float(group["eval_ifvar"].std(ddof=1)) if len(group) > 1 else 0.0,
                "eval_ifvar_ratio_vs_zero": float(group["eval_ifvar"].mean() / baseline_ifvar),
                "eval_mse_mean": float(group["eval_mse"].mean()),
                "eval_mse_ratio_vs_zero": float(group["eval_mse"].mean() / baseline_mse),
                "epochs_run_mean": float(group["epochs_run"].mean()),
                "seconds_mean": float(group["seconds"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    budget_rows: list[dict] = []
    for _, row in summary.iterrows():
        if row["method"] == "zero_surrogate":
            continue
        s = int(row["s"])
        for budget in DEFAULT_BUDGETS:
            if s >= budget:
                continue
            ratio = float((row["eval_ifvar_mean"] / (budget - s)) / (baseline_ifvar / budget))
            budget_rows.append(
                {
                    "method": row["method"],
                    "s": s,
                    "budget": budget,
                    "variance_ratio_vs_direct": ratio,
                    "wins_direct": ratio < 1.0,
                }
            )
    budget = pd.DataFrame(budget_rows)
    return result, summary, budget


def write_report(
    path: str | Path,
    *,
    embedding_npz: str | Path,
    target: str,
    summary: pd.DataFrame,
    budget: pd.DataFrame,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    best = summary.loc[summary["method"] != "zero_surrogate"].sort_values("eval_ifvar_mean").head(10)
    budget_best = (
        budget.sort_values(["budget", "variance_ratio_vs_direct"])
        .groupby("budget", as_index=False)
        .head(3)
        if not budget.empty
        else budget
    )
    lines = [
        "# HelpSteer2 Frozen-Embedding MLP Scaling Pilot",
        "",
        f"Embedding NPZ: `{embedding_npz}`",
        f"Target coefficient: `{target}`",
        "",
        "## Best IFVar Settings",
        "",
        best.to_markdown(index=False, floatfmt=".4f") if not best.empty else "(no trained models)",
        "",
        "## Direct-Regression Budget Comparison",
        "",
        budget_best.to_markdown(index=False, floatfmt=".4f") if not budget_best.empty else "(no budget rows)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HelpSteer2 frozen-embedding MLP scaling-law pilot.")
    parser.add_argument("--input-csv", default="Data/helpsteer2_preference_pairs.csv")
    parser.add_argument("--embedding-npz", required=True)
    parser.add_argument("--output-dir", default="artifacts/helpsteer2_preference/mlp_scaling")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    parser.add_argument("--s-grid", default="0,50,100,250,500,750,1000,1500,3000")
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--methods", nargs="*", default=[method.method for method in DEFAULT_METHODS])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--hessian-ridge", type=float, default=0.0)
    parser.add_argument(
        "--no-antisymmetric",
        action="store_true",
        help="Disable anti-symmetric prediction. Default is yhat(h) = (g(h)-g(-h))/2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    embeddings = load_embeddings(args.embedding_npz)["pair_embedding"].astype(np.float32)
    methods = parse_methods(args.methods)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result, summary, budget = run_scaling_experiment(
        frame,
        embeddings,
        target=args.target,
        feature_columns=args.features,
        s_grid=parse_int_list(args.s_grid),
        replications=args.replications,
        seed=args.seed,
        methods=methods,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        hessian_ridge=args.hessian_ridge,
        antisymmetric=not args.no_antisymmetric,
    )

    result_path = output_dir / "mlp_scaling_results.csv"
    summary_path = output_dir / "mlp_scaling_summary.csv"
    budget_path = output_dir / "budget_comparison.csv"
    report_path = output_dir / "mlp_scaling_report.md"
    metadata_path = output_dir / "metadata.json"
    result.to_csv(result_path, index=False)
    summary.to_csv(summary_path, index=False)
    budget.to_csv(budget_path, index=False)
    write_report(report_path, embedding_npz=args.embedding_npz, target=args.target, summary=summary, budget=budget)
    write_json(
        metadata_path,
        {
            "input_csv": args.input_csv,
            "embedding_npz": args.embedding_npz,
            "target": args.target,
            "features": args.features,
            "s_grid": parse_int_list(args.s_grid),
            "replications": args.replications,
            "methods": [method.__dict__ for method in methods],
            "hidden_dim": args.hidden_dim,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "antisymmetric": not args.no_antisymmetric,
        },
    )
    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {budget_path}")
    print(f"wrote {report_path}")
    print(summary.sort_values(["method", "s"]).to_string(index=False))


if __name__ == "__main__":
    main()
