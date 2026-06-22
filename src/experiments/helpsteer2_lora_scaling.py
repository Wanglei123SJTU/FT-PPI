from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.formatting import dataframe_to_markdown


Y_COL = "y_preference_strength"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_FEATURES = ["delta_log_length_scale", "delta_log_sentences_scale"]
DEFAULT_TARGETS = ["delta_log_length_scale"]
DEFAULT_S_GRID = [50, 100, 150, 200, 300, 400, 500, 700]


@dataclass(frozen=True)
class MethodSpec:
    method: str
    objective: str
    stop_metric: str
    label: str


DEFAULT_METHODS = [
    MethodSpec("mse_stop_mse", "mse", "mse", "MSE loss, stop by MSE"),
    MethodSpec("mse_stop_ifvar", "mse", "ifvar", "MSE loss, stop by Var"),
    MethodSpec("ifvar_stop_ifvar", "ifvar", "ifvar", "Var loss, stop by Var"),
]


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def parse_csv_list(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def parse_int_list(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def method_specs(names: Iterable[str] | None = None) -> list[MethodSpec]:
    specs = {item.method: item for item in DEFAULT_METHODS}
    if names is None:
        return list(DEFAULT_METHODS)
    selected = []
    for name in names:
        if name not in specs:
            raise ValueError(f"unknown method {name!r}; allowed={sorted(specs)}")
        selected.append(specs[name])
    return selected


def normalize_helpsteer_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    source_columns = ["delta_log_length", "delta_log_sentences"]
    for column in source_columns:
        scaled = f"{column}_scale"
        if scaled in out.columns:
            continue
        if column not in out.columns:
            raise ValueError(f"missing required feature column: {column}")
        sd = float(np.std(out[column].astype(float).to_numpy(), ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"cannot scale {column}; sd={sd}")
        out[scaled] = out[column].astype(float) / sd
    return out


def format_pair_text(prompt: str, response_1: str, response_2: str) -> str:
    return (
        "Task: Predict the human preference strength for Candidate B over Candidate A.\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Candidate A:\n{response_1}\n\n"
        f"Candidate B:\n{response_2}\n\n"
        "Preference strength for Candidate B over Candidate A:"
    )


def build_pair_texts(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    required = {"prompt", "response_1", "response_2"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing pair text columns: {missing}")
    forward = [
        format_pair_text(prompt, response_1, response_2)
        for prompt, response_1, response_2 in zip(frame["prompt"], frame["response_1"], frame["response_2"])
    ]
    swapped = [
        format_pair_text(prompt, response_2, response_1)
        for prompt, response_1, response_2 in zip(frame["prompt"], frame["response_1"], frame["response_2"])
    ]
    return forward, swapped


def design_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return np.column_stack([np.ones(len(frame)), frame[feature_columns].astype(float).to_numpy()])


def compute_ols_and_if_weights(
    frame: pd.DataFrame,
    *,
    target: str,
    feature_columns: list[str],
    hessian_ridge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target not in feature_columns:
        raise ValueError(f"target={target!r} must be in feature_columns={feature_columns}")
    y = frame[Y_COL].astype(float).to_numpy()
    x = design_matrix(frame, feature_columns)
    hessian = x.T @ x / len(x)
    if hessian_ridge > 0:
        hessian = hessian + hessian_ridge * np.eye(hessian.shape[0])
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    h_inv = np.linalg.pinv(hessian)
    target_idx = feature_columns.index(target) + 1
    if_weights = x @ h_inv[target_idx, :]
    return beta, hessian, if_weights


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
    n_validation = int(round(validation_frac * n))
    return {
        "train_pool": np.sort(perm[:n_train]),
        "validation": np.sort(perm[n_train : n_train + n_validation]),
        "evaluation": np.sort(perm[n_train + n_validation :]),
    }


def nested_train_indices(train_pool: np.ndarray, *, seed: int, replication: int, s: int) -> np.ndarray:
    if s > len(train_pool):
        raise ValueError(f"s={s} exceeds train pool size={len(train_pool)}")
    rng = np.random.default_rng(seed + 1_000_003 * (replication + 1))
    order = rng.permutation(train_pool)
    return np.sort(order[:s])


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, if_weights: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    if_residual = if_weights * residual
    return {
        "mse": float(np.mean(residual**2)),
        "ifvar": float(np.var(if_residual, ddof=0)),
        "ifmean": float(np.mean(if_residual)),
    }


def baseline_metrics(frame: pd.DataFrame, if_weights: np.ndarray, evaluation_idx: np.ndarray) -> dict[str, float]:
    y = frame[Y_COL].astype(float).to_numpy()
    pred = np.zeros(len(frame), dtype=float)
    return evaluate_predictions(y[evaluation_idx], pred[evaluation_idx], if_weights[evaluation_idx])


def make_cell_plan(
    *,
    targets: list[str],
    methods: list[MethodSpec],
    s_grid: list[int],
    replications: int,
) -> pd.DataFrame:
    rows = []
    for target in targets:
        for replication in range(replications):
            for s in s_grid:
                for method in methods:
                    rows.append(
                        {
                            "target": target,
                            "replication": replication,
                            "s": int(s),
                            "method": method.method,
                            "objective": method.objective,
                            "stop_metric": method.stop_metric,
                            "method_label": method.label,
                        }
                    )
    plan = pd.DataFrame(rows)
    plan.insert(0, "task_index", np.arange(len(plan), dtype=int))
    return plan


def _load_tokenizer_and_model(
    *,
    model_name: str,
    load_in_4bit: bool,
    dtype: str,
    trust_remote_code: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    gradient_checkpointing: bool,
):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs = {
        "num_labels": 1,
        "problem_type": "regression",
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype_map[dtype]
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    elif gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=["score"],
    )
    model = get_peft_model(model, config)
    model.train()
    return tokenizer, model


def _tokenize_texts(tokenizer, texts: list[str], *, max_length: int):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def _batch_iter(indices: np.ndarray, *, batch_size: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    shuffled = rng.permutation(indices)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start : start + batch_size]


def _move_batch(encoded: dict, idx: np.ndarray, device):
    import torch

    tensor_idx = torch.as_tensor(idx, dtype=torch.long)
    return {key: value[tensor_idx].to(device) for key, value in encoded.items()}


def _predict_batch(model, encoded_forward: dict, encoded_swapped: dict) -> np.ndarray:
    import torch

    with torch.inference_mode():
        forward = model(**encoded_forward).logits.reshape(-1)
        swapped = model(**encoded_swapped).logits.reshape(-1)
        pred = 0.5 * (forward - swapped)
    return pred.detach().float().cpu().numpy()


def _predict_indices(
    model,
    encoded_forward: dict,
    encoded_swapped: dict,
    indices: np.ndarray,
    *,
    batch_size: int,
    device,
) -> np.ndarray:
    values = []
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        batch_forward = _move_batch(encoded_forward, idx, device)
        batch_swapped = _move_batch(encoded_swapped, idx, device)
        values.append(_predict_batch(model, batch_forward, batch_swapped))
    return np.concatenate(values) if values else np.array([], dtype=float)


def train_lora_cell(
    frame: pd.DataFrame,
    *,
    target: str,
    feature_columns: list[str],
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    evaluation_idx: np.ndarray,
    method: MethodSpec,
    seed: int,
    model_name: str,
    load_in_4bit: bool,
    dtype: str,
    trust_remote_code: bool,
    max_length: int,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    gradient_checkpointing: bool,
    hessian_ridge: float,
) -> dict[str, float | str | int]:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(
        "train_lora_cell_start "
        f"target={target} method={method.method} seed={seed} train_size={len(train_idx)}",
        flush=True,
    )

    beta, hessian, if_weights = compute_ols_and_if_weights(
        frame,
        target=target,
        feature_columns=feature_columns,
        hessian_ridge=hessian_ridge,
    )
    print(
        "computed_if_weights "
        f"hessian_condition={np.linalg.cond(hessian):.4f} baseline_eval_ifvar="
        f"{baseline_metrics(frame, if_weights, evaluation_idx)['ifvar']:.6f}",
        flush=True,
    )
    baseline_eval = baseline_metrics(frame, if_weights, evaluation_idx)
    forward_texts, swapped_texts = build_pair_texts(frame)
    print(f"loading_model model_name={model_name}", flush=True)
    tokenizer, model = _load_tokenizer_and_model(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        gradient_checkpointing=gradient_checkpointing,
    )
    device = next(model.parameters()).device
    print(f"model_loaded device={device}", flush=True)
    print(f"tokenizing n_texts={len(forward_texts)} max_length={max_length}", flush=True)
    encoded_forward = _tokenize_texts(tokenizer, forward_texts, max_length=max_length)
    encoded_swapped = _tokenize_texts(tokenizer, swapped_texts, max_length=max_length)
    print("tokenized", flush=True)

    y = frame[Y_COL].astype(float).to_numpy(dtype=np.float32)
    y_tensor = torch.from_numpy(y).to(device)
    if_tensor = torch.from_numpy(if_weights.astype(np.float32)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = None
    best_metric = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    rng = np.random.default_rng(seed)

    for epoch in range(max_epochs):
        print(f"epoch_start epoch={epoch + 1}/{max_epochs}", flush=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_count = 0
        for batch_number, batch_idx in enumerate(_batch_iter(train_idx, batch_size=train_batch_size, rng=rng), start=1):
            batch_forward = _move_batch(encoded_forward, batch_idx, device)
            batch_swapped = _move_batch(encoded_swapped, batch_idx, device)
            forward = model(**batch_forward).logits.reshape(-1)
            swapped = model(**batch_swapped).logits.reshape(-1)
            pred = 0.5 * (forward - swapped)
            residual = y_tensor[torch.as_tensor(batch_idx, dtype=torch.long, device=device)] - pred
            if method.objective == "mse":
                loss = torch.mean(residual**2)
            elif method.objective == "ifvar":
                a_batch = if_tensor[torch.as_tensor(batch_idx, dtype=torch.long, device=device)]
                if_residual = a_batch * residual
                loss = torch.mean((if_residual - torch.mean(if_residual)) ** 2)
            else:
                raise ValueError(f"unknown objective={method.objective}")
            loss = loss / max(1, gradient_accumulation_steps)
            loss.backward()
            if batch_number % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1
        if step_count == 0 or batch_number % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        model.eval()
        val_pred = _predict_indices(
            model,
            encoded_forward,
            encoded_swapped,
            validation_idx,
            batch_size=eval_batch_size,
            device=device,
        )
        val_metrics = evaluate_predictions(y[validation_idx], val_pred, if_weights[validation_idx])
        metric = val_metrics["mse"] if method.stop_metric == "mse" else val_metrics["ifvar"]
        print(
            "epoch_done "
            f"epoch={epoch + 1} val_mse={val_metrics['mse']:.6f} "
            f"val_ifvar={val_metrics['ifvar']:.6f} stop_metric={metric:.6f}",
            flush=True,
        )
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
    eval_pred = _predict_indices(
        model,
        encoded_forward,
        encoded_swapped,
        evaluation_idx,
        batch_size=eval_batch_size,
        device=device,
    )
    val_pred = _predict_indices(
        model,
        encoded_forward,
        encoded_swapped,
        validation_idx,
        batch_size=eval_batch_size,
        device=device,
    )
    eval_metrics = evaluate_predictions(y[evaluation_idx], eval_pred, if_weights[evaluation_idx])
    val_metrics = evaluate_predictions(y[validation_idx], val_pred, if_weights[validation_idx])
    print(
        "train_lora_cell_done "
        f"target={target} method={method.method} train_size={len(train_idx)} "
        f"eval_ifvar={eval_metrics['ifvar']:.6f} eval_mse={eval_metrics['mse']:.6f}",
        flush=True,
    )

    result = {
        "target": target,
        "method": method.method,
        "objective": method.objective,
        "stop_metric": method.stop_metric,
        "method_label": method.label,
        "train_size": int(len(train_idx)),
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch + 1),
        "best_validation_metric": float(best_metric),
        "eval_mse": eval_metrics["mse"],
        "eval_ifvar": eval_metrics["ifvar"],
        "eval_ifmean": eval_metrics["ifmean"],
        "val_mse": val_metrics["mse"],
        "val_ifvar": val_metrics["ifvar"],
        "baseline_eval_mse": baseline_eval["mse"],
        "baseline_eval_ifvar": baseline_eval["ifvar"],
        "baseline_eval_ifmean": baseline_eval["ifmean"],
        "beta_intercept": float(beta[0]),
        "beta_target": float(beta[feature_columns.index(target) + 1]),
        "hessian_condition": float(np.linalg.cond(hessian)),
    }
    del model, tokenizer, encoded_forward, encoded_swapped
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def summarize_results(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in cells.groupby(["target", "method", "method_label", "s"], sort=True):
        target, method, method_label, s = keys
        replications = int(len(group))
        sd = float(group["eval_ifvar"].std(ddof=1)) if replications > 1 else 0.0
        se = sd / math.sqrt(replications) if replications > 0 else math.nan
        rows.append(
            {
                "target": target,
                "method": method,
                "method_label": method_label,
                "s": int(s),
                "replications": replications,
                "eval_ifvar_mean": float(group["eval_ifvar"].mean()),
                "eval_ifvar_sd": sd,
                "eval_ifvar_se": se,
                "eval_ifvar_ci_low": float(group["eval_ifvar"].mean() - 1.96 * se),
                "eval_ifvar_ci_high": float(group["eval_ifvar"].mean() + 1.96 * se),
                "eval_mse_mean": float(group["eval_mse"].mean()),
                "val_ifvar_mean": float(group["val_ifvar"].mean()),
                "val_mse_mean": float(group["val_mse"].mean()),
                "baseline_eval_ifvar": float(group["baseline_eval_ifvar"].mean()),
                "baseline_eval_mse": float(group["baseline_eval_mse"].mean()),
                "best_epoch_mean": float(group["best_epoch"].mean()),
                "epochs_run_mean": float(group["epochs_run"].mean()),
            }
        )
    return pd.DataFrame(rows)


def fit_power_law(summary: pd.DataFrame) -> pd.DataFrame:
    from scipy.optimize import curve_fit

    def curve(s, a, alpha, b):
        return a * np.power(s, -alpha) + b

    rows = []
    for keys, group in summary.groupby(["target", "method", "method_label"], sort=True):
        target, method, method_label = keys
        clean = group.loc[(group["s"] > 0) & np.isfinite(group["eval_ifvar_mean"])].sort_values("s")
        if len(clean) < 4:
            continue
        s = clean["s"].to_numpy(dtype=float)
        y = clean["eval_ifvar_mean"].to_numpy(dtype=float)
        b0 = max(0.0, float(np.min(y) * 0.8))
        a0 = max(1e-6, float((np.max(y) - b0) * (np.min(s) ** 0.5)))
        try:
            popt, _ = curve_fit(
                curve,
                s,
                y,
                p0=(a0, 0.5, b0),
                bounds=([0.0, 0.01, 0.0], [np.inf, 3.0, np.inf]),
                maxfev=100_000,
            )
            pred = curve(s, *popt)
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
            rows.append(
                {
                    "target": target,
                    "method": method,
                    "method_label": method_label,
                    "a_hat": float(popt[0]),
                    "alpha_hat": float(popt[1]),
                    "b_hat": float(popt[2]),
                    "r2": float(r2),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "target": target,
                    "method": method,
                    "method_label": method_label,
                    "a_hat": math.nan,
                    "alpha_hat": math.nan,
                    "b_hat": math.nan,
                    "r2": math.nan,
                    "fit_error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def plot_scaling(summary: pd.DataFrame, fit: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = {
        "mse_stop_mse": "#1f77b4",
        "mse_stop_ifvar": "#ff7f0e",
        "ifvar_stop_ifvar": "#2ca02c",
    }
    markers = {
        "mse_stop_mse": "o",
        "mse_stop_ifvar": "s",
        "ifvar_stop_ifvar": "^",
    }
    for target, target_df in summary.groupby("target", sort=True):
        baseline = float(target_df["baseline_eval_ifvar"].mean())
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        for method, method_df in target_df.groupby("method", sort=False):
            method_df = method_df.sort_values("s")
            label = str(method_df["method_label"].iloc[0])
            x = method_df["s"].to_numpy(dtype=float)
            y = method_df["eval_ifvar_mean"].to_numpy(dtype=float)
            low = method_df["eval_ifvar_ci_low"].to_numpy(dtype=float)
            high = method_df["eval_ifvar_ci_high"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                label=label,
                color=colors.get(method),
                marker=markers.get(method, "o"),
                linewidth=2,
                markersize=6,
            )
            if np.any(high > low):
                ax.fill_between(x, low, high, color=colors.get(method), alpha=0.12, linewidth=0)
        ax.axhline(baseline, color="#aa3377", linestyle="--", linewidth=1.8)
        ax.text(0.98, baseline, f"Baseline Var = {baseline:.3f}", color="#aa3377", ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.set_xlabel("FT subset size (s)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Target IF residual variance", fontsize=13, fontweight="bold")
        ax.set_title(f"HelpSteer2 LoRA scaling: {target}", fontsize=13)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            path = figure_dir / f"helpsteer2_lora_scaling_{target}.{ext}"
            fig.savefig(path, dpi=180 if ext == "png" else None)
            paths.append(path)
        plt.close(fig)
    return paths


def write_report(
    path: str | Path,
    *,
    summary: pd.DataFrame,
    fit: pd.DataFrame,
    metadata: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fit_display = fit[["target", "method_label", "a_hat", "alpha_hat", "b_hat", "r2"]] if not fit.empty else fit
    best = summary.sort_values(["target", "eval_ifvar_mean"]).groupby("target", as_index=False).head(10)
    lines = [
        "# HelpSteer2 Qwen LoRA Scaling Pilot",
        "",
        "## Setup",
        "",
        f"- Model: `{metadata['model_name']}`",
        f"- Features: `{', '.join(metadata['feature_columns'])}`",
        f"- Targets: `{', '.join(metadata['targets'])}`",
        f"- Methods: `{', '.join(metadata['methods'])}`",
        f"- s grid: `{metadata['s_grid']}`",
        f"- Replications: `{metadata['replications']}`",
        "",
        "## Best Observed Cells",
        "",
        dataframe_to_markdown(best, index=False, floatfmt=".4f") if not best.empty else "(no completed cells)",
        "",
        "## Scaling-Law Fit",
        "",
        dataframe_to_markdown(fit_display, index=False, floatfmt=".4f") if not fit_display.empty else "(not enough cells)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def describe_command(args: argparse.Namespace) -> None:
    frame = normalize_helpsteer_features(pd.read_csv(args.input_csv))
    feature_columns = parse_csv_list(args.features)
    targets = parse_csv_list(args.targets)
    rows = []
    for target in targets:
        beta, hessian, if_weights = compute_ols_and_if_weights(
            frame,
            target=target,
            feature_columns=feature_columns,
            hessian_ridge=args.hessian_ridge,
        )
        rows.append(
            {
                "target": target,
                "beta_target": float(beta[feature_columns.index(target) + 1]),
                "hessian_condition": float(np.linalg.cond(hessian)),
                "if_weight_p50_abs": float(np.quantile(np.abs(if_weights), 0.50)),
                "if_weight_p90_abs": float(np.quantile(np.abs(if_weights), 0.90)),
                "if_weight_p95_abs": float(np.quantile(np.abs(if_weights), 0.95)),
                "if_weight_p99_abs": float(np.quantile(np.abs(if_weights), 0.99)),
                "if_weight_max_abs": float(np.max(np.abs(if_weights))),
            }
        )
    out = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "target_diagnostics.csv", index=False)
    print(dataframe_to_markdown(out, index=False, floatfmt=".4f"))


def make_plan_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = parse_csv_list(args.targets)
    methods = method_specs(parse_csv_list(args.methods))
    s_grid = parse_int_list(args.s_grid)
    plan = make_cell_plan(targets=targets, methods=methods, s_grid=s_grid, replications=args.replications)
    plan.to_csv(output_dir / "cell_plan.csv", index=False)
    metadata = vars(args).copy()
    metadata["targets"] = targets
    metadata["feature_columns"] = parse_csv_list(args.features)
    metadata["s_grid"] = s_grid
    metadata["methods"] = [method.method for method in methods]
    metadata["n_cells"] = int(len(plan))
    write_json(output_dir / "plan_metadata.json", metadata)
    print(f"wrote {output_dir / 'cell_plan.csv'} n_cells={len(plan)}")


def train_cell_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    plan = pd.read_csv(args.plan_csv)
    row = plan.loc[plan["task_index"].astype(int) == int(args.task_index)]
    if row.empty:
        raise ValueError(f"task_index={args.task_index} not found in {args.plan_csv}")
    cell = row.iloc[0].to_dict()
    method = MethodSpec(
        method=str(cell["method"]),
        objective=str(cell["objective"]),
        stop_metric=str(cell["stop_metric"]),
        label=str(cell["method_label"]),
    )
    frame = normalize_helpsteer_features(pd.read_csv(args.input_csv))
    feature_columns = parse_csv_list(args.features)
    splits = split_indices(len(frame), seed=args.seed)
    train_idx = nested_train_indices(
        splits["train_pool"],
        seed=args.seed,
        replication=int(cell["replication"]),
        s=int(cell["s"]),
    )
    start = time.time()
    result = train_lora_cell(
        frame,
        target=str(cell["target"]),
        feature_columns=feature_columns,
        train_idx=train_idx,
        validation_idx=splits["validation"],
        evaluation_idx=splits["evaluation"],
        method=method,
        seed=args.seed + 17 * int(cell["task_index"]) + 101,
        model_name=args.model_name,
        load_in_4bit=not args.no_4bit,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=parse_csv_list(args.target_modules),
        gradient_checkpointing=not args.no_gradient_checkpointing,
        hessian_ridge=args.hessian_ridge,
    )
    result.update(
        {
            "task_index": int(cell["task_index"]),
            "replication": int(cell["replication"]),
            "s": int(cell["s"]),
            "seconds": time.time() - start,
            "model_name": args.model_name,
            "max_length": args.max_length,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
    )
    cell_dir = output_dir / "cells"
    cell_dir.mkdir(parents=True, exist_ok=True)
    path = cell_dir / f"cell_{int(cell['task_index']):04d}.json"
    write_json(path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def aggregate_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    cell_paths = sorted((output_dir / "cells").glob("cell_*.json"))
    if not cell_paths:
        raise ValueError(f"no cell outputs found under {output_dir / 'cells'}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in cell_paths]
    cells = pd.DataFrame(rows)
    cells.to_csv(output_dir / "lora_scaling_cells.csv", index=False)
    summary = summarize_results(cells)
    summary.to_csv(output_dir / "lora_scaling_summary.csv", index=False)
    fit = fit_power_law(summary)
    fit.to_csv(output_dir / "lora_scaling_fit_table.csv", index=False)
    plot_scaling(summary, fit, output_dir)
    metadata = {
        "model_name": cells["model_name"].iloc[0] if "model_name" in cells.columns else "unknown",
        "feature_columns": parse_csv_list(args.features),
        "targets": sorted(cells["target"].unique().tolist()),
        "methods": sorted(cells["method"].unique().tolist()),
        "s_grid": sorted(int(x) for x in cells["s"].unique()),
        "replications": int(cells["replication"].nunique()),
        "n_cells_completed": int(len(cells)),
    }
    write_json(output_dir / "aggregate_metadata.json", metadata)
    write_report(output_dir / "lora_scaling_report.md", summary=summary, fit=fit, metadata=metadata)
    print(f"aggregated cells={len(cells)}")
    print(dataframe_to_markdown(fit, index=False, floatfmt=".4f") if not fit.empty else "no fit rows")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HelpSteer2 Qwen LoRA scaling-law pilot.")
    parser.add_argument("--input-csv", default="Data/helpsteer2_preference_pairs.csv")
    parser.add_argument("--output-dir", default="artifacts/helpsteer2_preference/lora_scaling")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--s-grid", default=",".join(str(x) for x in DEFAULT_S_GRID))
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--methods", default=",".join(method.method for method in DEFAULT_METHODS))
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--hessian-ridge", type=float, default=0.0)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("describe")
    subparsers.add_parser("make-plan")

    train = subparsers.add_parser("train-cell")
    train.add_argument("--plan-csv", required=True)
    train.add_argument("--task-index", type=int, required=True)
    train.add_argument("--model-name", default=DEFAULT_MODEL)
    train.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    train.add_argument("--no-4bit", action="store_true")
    train.add_argument("--trust-remote-code", action="store_true")
    train.add_argument("--max-length", type=int, default=768)
    train.add_argument("--train-batch-size", type=int, default=4)
    train.add_argument("--eval-batch-size", type=int, default=8)
    train.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--max-epochs", type=int, default=8)
    train.add_argument("--patience", type=int, default=2)
    train.add_argument("--lora-r", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=32)
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    train.add_argument("--no-gradient-checkpointing", action="store_true")

    subparsers.add_parser("aggregate")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "describe":
        describe_command(args)
    elif args.command == "make-plan":
        make_plan_command(args)
    elif args.command == "train-cell":
        train_cell_command(args)
    elif args.command == "aggregate":
        aggregate_command(args)
    else:
        raise ValueError(f"unknown command={args.command}")


if __name__ == "__main__":
    main()
