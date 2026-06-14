from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


TEXT_COL = "description"
Y_COL = "points"


@dataclass(frozen=True)
class SplitBundle:
    l_ids: np.ndarray
    v_stop_ids: np.ndarray
    v_scale_ids: np.ndarray
    l_prime_ids: np.ndarray
    train_order_ids: np.ndarray
    e_eval_ids: np.ndarray


@dataclass(frozen=True)
class MethodSpec:
    name: str
    training_loss: str
    early_stopping_metric: str


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_wine(input_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    missing = {TEXT_COL, Y_COL} - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    clean = df[[TEXT_COL, Y_COL]].dropna().drop_duplicates(TEXT_COL, keep="first").copy()
    clean[Y_COL] = clean[Y_COL].astype(float)
    clean = clean.reset_index(drop=True)
    clean.insert(0, "sample_id", np.arange(len(clean), dtype=np.int64))
    return clean


def scaled_y(y_raw: pd.Series | np.ndarray) -> np.ndarray:
    return (np.asarray(y_raw, dtype=float) - 90.0) / 5.0


def raw_y(y_scaled: pd.Series | np.ndarray) -> np.ndarray:
    return 90.0 + 5.0 * np.asarray(y_scaled, dtype=float)


def _rep_seed(config: dict[str, Any], replication_id: int, salt: int = 0) -> int:
    base_seed = int(config.get("seed", 20260613))
    return base_seed + 100_003 * int(replication_id) + int(salt)


def validation_stop_size(config: dict[str, Any]) -> int:
    return int(config.get("validation_stop_size", config.get("validation_size", 0)))


def validation_scale_size(config: dict[str, Any]) -> int:
    return int(config.get("validation_scale_size", 0))


def n_effective_labeled(config: dict[str, Any]) -> int:
    return int(config["budget_B"]) - validation_stop_size(config) - validation_scale_size(config)


def configured_losses(config: dict[str, Any]) -> list[str]:
    raw_losses = config.get("losses", [config.get("loss", "var")])
    if isinstance(raw_losses, str):
        raw_losses = [raw_losses]
    losses = [str(loss).lower() for loss in raw_losses]
    allowed = {"var", "mse"}
    unknown = sorted(set(losses) - allowed)
    if unknown:
        raise ValueError(f"unsupported losses: {unknown}; expected subset of {sorted(allowed)}")
    if not losses:
        raise ValueError("at least one loss is required")
    return losses


def _validate_method_name(name: str) -> str:
    clean = str(name).lower()
    allowed_chars = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    if not clean or any(ch not in allowed_chars for ch in clean):
        raise ValueError(f"invalid method name {name!r}; use lowercase letters, digits, '_' or '-'")
    return clean


def configured_methods(config: dict[str, Any]) -> list[MethodSpec]:
    raw_methods = config.get("methods")
    if raw_methods is None:
        stop_metric = str(config.get("early_stopping_metric", "var")).lower()
        if stop_metric not in {"var", "mse"}:
            raise ValueError("early_stopping_metric must be 'var' or 'mse'")
        return [
            MethodSpec(name=loss_name, training_loss=loss_name, early_stopping_metric=stop_metric)
            for loss_name in configured_losses(config)
        ]
    if not isinstance(raw_methods, list) or not raw_methods:
        raise ValueError("methods must be a non-empty list")

    methods: list[MethodSpec] = []
    seen: set[str] = set()
    for raw in raw_methods:
        if not isinstance(raw, dict):
            raise ValueError("each method entry must be a mapping")
        training_loss = str(raw.get("loss", raw.get("training_loss", ""))).lower()
        if training_loss not in {"var", "mse"}:
            raise ValueError(f"unsupported method loss {training_loss!r}; expected 'var' or 'mse'")
        stop_metric = str(raw.get("early_stopping_metric", "var")).lower()
        if stop_metric not in {"var", "mse"}:
            raise ValueError(f"unsupported early_stopping_metric {stop_metric!r}; expected 'var' or 'mse'")
        name = _validate_method_name(raw.get("name", f"{training_loss}_stop_{stop_metric}"))
        if name in seen:
            raise ValueError(f"duplicate method name: {name}")
        seen.add(name)
        methods.append(MethodSpec(name=name, training_loss=training_loss, early_stopping_metric=stop_metric))
    return methods


def method_by_name(config: dict[str, Any], method_name: str) -> MethodSpec:
    name = _validate_method_name(method_name)
    methods = {method.name: method for method in configured_methods(config)}
    if name not in methods:
        raise ValueError(f"method {name!r} is not listed in config methods")
    return methods[name]


def build_replication_splits(clean: pd.DataFrame, config: dict[str, Any], replication_id: int) -> SplitBundle:
    l_size = int(config["budget_B"])
    v_stop_size = validation_stop_size(config)
    v_scale_size = validation_scale_size(config)
    e_size = int(config["eval_size"])
    s_grid = [int(x) for x in config["s_grid"]]
    if max(s_grid) > l_size - v_stop_size - v_scale_size:
        raise ValueError("largest train size exceeds |L \\ (V_stop union V_scale)|")
    if l_size + e_size > len(clean):
        raise ValueError(f"need at least {l_size + e_size} clean rows, found {len(clean)}")

    rng = np.random.default_rng(_rep_seed(config, replication_id))
    all_ids = clean["sample_id"].to_numpy(dtype=np.int64)
    l_ids = rng.choice(all_ids, size=l_size, replace=False)
    v_stop_ids = rng.choice(l_ids, size=v_stop_size, replace=False)
    remaining_after_stop = np.setdiff1d(l_ids, v_stop_ids, assume_unique=False)
    v_scale_ids = rng.choice(remaining_after_stop, size=v_scale_size, replace=False)
    l_prime_ids = np.setdiff1d(remaining_after_stop, v_scale_ids, assume_unique=False)
    train_order_ids = rng.permutation(l_prime_ids)
    outside_l = np.setdiff1d(all_ids, l_ids, assume_unique=False)
    e_eval_ids = rng.choice(outside_l, size=e_size, replace=False)
    bundle = SplitBundle(
        l_ids=np.asarray(l_ids, dtype=np.int64),
        v_stop_ids=np.asarray(v_stop_ids, dtype=np.int64),
        v_scale_ids=np.asarray(v_scale_ids, dtype=np.int64),
        l_prime_ids=np.asarray(l_prime_ids, dtype=np.int64),
        train_order_ids=np.asarray(train_order_ids, dtype=np.int64),
        e_eval_ids=np.asarray(e_eval_ids, dtype=np.int64),
    )
    validate_split_bundle(bundle, s_grid)
    return bundle


def train_ids_for_s(bundle: SplitBundle, s_train: int) -> np.ndarray:
    if s_train > len(bundle.train_order_ids):
        raise ValueError("s_train exceeds nested training order length")
    return bundle.train_order_ids[: int(s_train)]


def ids_hash(ids: Iterable[int]) -> str:
    arr = np.asarray(list(ids), dtype=np.int64)
    arr = np.sort(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def validate_split_bundle(bundle: SplitBundle, s_grid: list[int]) -> list[str]:
    failures: list[str] = []
    l = set(map(int, bundle.l_ids))
    v_stop = set(map(int, bundle.v_stop_ids))
    v_scale = set(map(int, bundle.v_scale_ids))
    l_prime = set(map(int, bundle.l_prime_ids))
    e_eval = set(map(int, bundle.e_eval_ids))
    if len(l) != len(bundle.l_ids):
        failures.append("L has duplicate sample_ids")
    if len(v_stop) != len(bundle.v_stop_ids):
        failures.append("V_stop has duplicate sample_ids")
    if len(v_scale) != len(bundle.v_scale_ids):
        failures.append("V_scale has duplicate sample_ids")
    if len(e_eval) != len(bundle.e_eval_ids):
        failures.append("E_eval has duplicate sample_ids")
    if not v_stop.issubset(l):
        failures.append("V_stop is not a subset of L")
    if not v_scale.issubset(l):
        failures.append("V_scale is not a subset of L")
    if v_stop & v_scale:
        failures.append("V_stop overlaps V_scale")
    if l_prime != l - v_stop - v_scale:
        failures.append("L_prime is not exactly L minus V_stop and V_scale")
    if e_eval & l:
        failures.append("E_eval overlaps L")
    previous: set[int] = set()
    for s in s_grid:
        current = set(map(int, train_ids_for_s(bundle, int(s))))
        if not current.issubset(l_prime):
            failures.append(f"train set s={s} is not a subset of L_prime")
        if not previous.issubset(current):
            failures.append(f"train set s={s} is not nested")
        previous = current
    if failures:
        raise ValueError("; ".join(failures))
    return failures


def subset_by_ids(clean: pd.DataFrame, ids: np.ndarray, role: str) -> pd.DataFrame:
    ids_df = pd.DataFrame({"sample_id": np.asarray(ids, dtype=np.int64), "_order": np.arange(len(ids))})
    out = ids_df.merge(clean, on="sample_id", how="left", validate="one_to_one").sort_values("_order")
    if out[TEXT_COL].isna().any():
        raise ValueError(f"{role} contains unknown sample_ids")
    out = out.drop(columns=["_order"]).reset_index(drop=True)
    out["split_role"] = role
    out["y_raw"] = out[Y_COL].astype(float)
    out["y_scaled"] = scaled_y(out["y_raw"])
    return out


def var_loss_from_residuals(residual: Any) -> Any:
    return ((residual - residual.mean()) ** 2).mean()


def mse_loss_from_residuals(residual: Any) -> Any:
    return (residual**2).mean()


def training_loss_from_residuals(residual: Any, loss_name: str) -> Any:
    loss_name = str(loss_name).lower()
    if loss_name == "var":
        return var_loss_from_residuals(residual)
    if loss_name == "mse":
        return mse_loss_from_residuals(residual)
    raise ValueError(f"unsupported loss: {loss_name}")


def stopping_value_from_metrics(metrics: dict[str, float], early_stopping_metric: str) -> float:
    metric = str(early_stopping_metric).lower()
    if metric == "var":
        return float(metrics["residual_var_scaled"])
    if metric == "mse":
        return float(metrics["rmse_scaled"]) ** 2
    raise ValueError(f"unsupported early stopping metric: {early_stopping_metric}")


def steps_per_epoch_for_s(s_train: int, batch_size: int) -> int:
    return max(1, math.ceil(int(s_train) / int(batch_size)))


def max_steps_for_s(s_train: int, batch_size: int, max_epochs: int = 12) -> int:
    return int(max_epochs) * steps_per_epoch_for_s(s_train, batch_size)


def is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda out of memory" in message or "cublas_status_alloc_failed" in message


def _require_training_modules():
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("training requires torch, transformers, peft, accelerate, pandas, pyarrow, and pyyaml") from exc
    return {
        "torch": torch,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "AutoModel": AutoModel,
        "AutoTokenizer": AutoTokenizer,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
    }


def set_training_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


class WineTokenizedDataset:
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int):
        self.sample_ids = frame["sample_id"].astype(int).tolist()
        self.y_raw = frame["y_raw"].astype(float).to_numpy()
        self.y_scaled = frame["y_scaled"].astype(float).to_numpy()
        self.encodings = tokenizer(
            frame[TEXT_COL].astype(str).tolist(),
            truncation=True,
            max_length=int(max_length),
            padding=False,
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "sample_id": self.sample_ids[idx],
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "y_raw": float(self.y_raw[idx]),
            "y_scaled": float(self.y_scaled[idx]),
        }


def make_collate_fn(tokenizer: Any, torch: Any):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        tokenized = tokenizer.pad(
            [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch],
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        tokenized["labels"] = torch.tensor([item["y_scaled"] for item in batch], dtype=torch.float32)
        tokenized["sample_id"] = torch.tensor([item["sample_id"] for item in batch], dtype=torch.long)
        tokenized["y_raw"] = torch.tensor([item["y_raw"] for item in batch], dtype=torch.float32)
        return tokenized

    return collate


def build_tokenizer(model_name: str, max_length: int, AutoTokenizer: Any) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_model(config: dict[str, Any], torch: Any, AutoModel: Any, LoraConfig: Any, get_peft_model: Any) -> Any:
    import torch.nn as nn

    class LastTokenRegressionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            backbone = AutoModel.from_pretrained(
                config["model_name"],
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            if hasattr(backbone.config, "use_cache"):
                backbone.config.use_cache = False
            if bool(config.get("gradient_checkpointing", False)) and hasattr(backbone, "gradient_checkpointing_enable"):
                backbone.gradient_checkpointing_enable()
            if bool(config.get("gradient_checkpointing", False)) and hasattr(backbone, "enable_input_require_grads"):
                backbone.enable_input_require_grads()
            lora_config = LoraConfig(
                r=int(config["lora_r"]),
                lora_alpha=int(config["lora_alpha"]),
                target_modules=config["target_modules"],
                lora_dropout=float(config["lora_dropout"]),
                bias="none",
            )
            self.backbone = get_peft_model(backbone, lora_config)
            hidden_size = int(backbone.config.hidden_size)
            self.head = nn.Linear(hidden_size, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
            return self.head(pooled.float()).squeeze(-1)

    model = LastTokenRegressionModel()
    return model


def _move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items() if key not in {"sample_id", "y_raw"}}


def capture_trainable_state(model: Any) -> dict[str, Any]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def restore_trainable_state(model: Any, state: dict[str, Any]) -> None:
    params = dict(model.named_parameters())
    for name, value in state.items():
        params[name].data.copy_(value.to(device=params[name].device, dtype=params[name].dtype))


def train_once(
    config: dict[str, Any],
    train_df: pd.DataFrame,
    stop_df: pd.DataFrame,
    scale_df: pd.DataFrame,
    eval_df: pd.DataFrame | None,
    batch_size: int,
    seed: int,
    loss_name: str,
    early_stopping_metric: str = "var",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    mods = _require_training_modules()
    torch = mods["torch"]
    set_training_seeds(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LoRA training")
    device = torch.device("cuda")

    tokenizer = build_tokenizer(config["model_name"], int(config["max_length"]), mods["AutoTokenizer"])
    model = build_model(config, torch, mods["AutoModel"], mods["LoraConfig"], mods["get_peft_model"]).to(device)
    model.train()
    try:
        model.backbone.print_trainable_parameters()
    except Exception:
        pass

    collate = make_collate_fn(tokenizer, torch)
    train_dataset = WineTokenizedDataset(train_df, tokenizer, int(config["max_length"]))
    train_loader = mods["DataLoader"](
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )
    max_epochs = int(config.get("max_epochs", 12))
    min_epochs = int(config.get("min_epochs", 4))
    patience = int(config.get("early_stopping_patience", 3))
    min_delta = float(config.get("early_stopping_min_delta", 0.0))
    steps_per_epoch = steps_per_epoch_for_s(len(train_df), batch_size)
    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": float(config["lora_lr"]), "weight_decay": float(config["weight_decay"])},
            {"params": head_params, "lr": float(config["head_lr"]), "weight_decay": float(config["weight_decay"])},
        ]
    )
    losses: list[float] = []
    epoch_history: list[dict[str, float | int | bool]] = []
    start = time.time()
    best_state: dict[str, Any] | None = None
    early_stopping_metric = str(early_stopping_metric).lower()
    best_stop_value = float("inf")
    best_stop_var = float("inf")
    best_stop_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    total_steps = 0
    early_stopped = False
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            inputs = _move_batch(batch, device)
            labels = inputs.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(**inputs)
            residual = labels.float() - pred.float()
            loss = training_loss_from_residuals(residual, loss_name)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            epoch_losses.append(loss_value)
            total_steps += 1

        stop_pred = predict_frame(model, tokenizer, stop_df, config, torch, mods["DataLoader"], batch_size)
        stop_metrics = prediction_metrics(stop_pred)
        stop_var = float(stop_metrics["residual_var_scaled"])
        stop_mse = float(stop_metrics["rmse_scaled"]) ** 2
        stop_value = stopping_value_from_metrics(stop_metrics, early_stopping_metric)
        improved = stop_value < best_stop_value - min_delta
        if improved:
            best_stop_value = stop_value
            best_stop_var = stop_var
            best_stop_mse = stop_mse
            best_epoch = epoch
            best_state = capture_trainable_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_history.append(
            {
                "epoch": int(epoch),
                "train_loss_mean": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                "train_loss_last": float(epoch_losses[-1]) if epoch_losses else float("nan"),
                "validation_stop_residual_var": stop_var,
                "validation_stop_mse": stop_mse,
                "early_stopping_metric_value": stop_value,
                "validation_stop_residual_mean": float(stop_metrics["residual_mean_scaled"]),
                "validation_stop_rmse": float(stop_metrics["rmse_scaled"]),
                "validation_stop_corr": float(stop_metrics["correlation"]),
                "is_best": bool(improved),
            }
        )
        if epoch % int(config.get("log_every_epochs", 1)) == 0 or epoch == max_epochs or improved:
            print(
                "train_progress",
                f"epoch={epoch}",
                f"max_epochs={max_epochs}",
                f"steps={total_steps}",
                f"loss={loss_name}",
                f"stop_metric={early_stopping_metric}",
                f"train_loss={epoch_history[-1]['train_loss_mean']:.6f}",
                f"v_stop_var={stop_var:.6f}",
                f"v_stop_mse={stop_mse:.6f}",
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            early_stopped = True
            print(
                "early_stop",
                f"epoch={epoch}",
                f"best_epoch={best_epoch}",
                f"stop_metric={early_stopping_metric}",
                f"best_value={best_stop_value:.6f}",
                f"best_v_stop_var={best_stop_var:.6f}",
                flush=True,
            )
            break

    if best_state is not None:
        restore_trainable_state(model, best_state)
    stop_pred = predict_frame(model, tokenizer, stop_df, config, torch, mods["DataLoader"], batch_size)
    scale_pred = predict_frame(model, tokenizer, scale_df, config, torch, mods["DataLoader"], batch_size)
    eval_pred = (
        predict_frame(model, tokenizer, eval_df, config, torch, mods["DataLoader"], batch_size)
        if eval_df is not None and len(eval_df) > 0
        else None
    )
    runtime = {
        "runtime_seconds": time.time() - start,
        "actual_batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "min_epochs": int(min_epochs),
        "early_stopping_patience": int(patience),
        "early_stopping_min_delta": float(min_delta),
        "early_stopping_metric": early_stopping_metric,
        "epochs_trained": int(epoch_history[-1]["epoch"]) if epoch_history else 0,
        "best_epoch": int(best_epoch),
        "early_stopped": bool(early_stopped),
        "steps_per_epoch": int(steps_per_epoch),
        "total_train_steps": int(total_steps),
        "max_steps": int(total_steps),
        "best_validation_stop_metric_value": float(best_stop_value),
        "best_validation_stop_residual_var": float(best_stop_var),
        "best_validation_stop_mse": float(best_stop_mse),
        "final_train_loss": float(losses[-1]) if losses else float("nan"),
        "mean_train_loss": float(np.mean(losses)) if losses else float("nan"),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "epoch_history": epoch_history,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return runtime, stop_pred, scale_pred, eval_pred


def predict_frame(
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    config: dict[str, Any],
    torch: Any,
    DataLoader: Any,
    batch_size: int,
) -> pd.DataFrame:
    model.eval()
    collate = make_collate_fn(tokenizer, torch)
    dataset = WineTokenizedDataset(frame, tokenizer, int(config["max_length"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("eval_batch_size", batch_size)),
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )
    rows: list[pd.DataFrame] = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            sample_ids = batch["sample_id"].cpu().numpy()
            y_raw_values = batch["y_raw"].cpu().numpy().astype(float)
            labels = batch["labels"].cpu().numpy().astype(float)
            inputs = _move_batch(batch, device)
            inputs.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_scaled = model(**inputs).float().detach().cpu().numpy().astype(float)
            rows.append(
                pd.DataFrame(
                    {
                        "sample_id": sample_ids.astype(np.int64),
                        "y_raw": y_raw_values,
                        "y_scaled": labels,
                        "pred_scaled": pred_scaled,
                        "pred_raw": raw_y(pred_scaled),
                    }
                )
            )
    out = pd.concat(rows, ignore_index=True)
    out["residual_scaled"] = out["y_scaled"] - out["pred_scaled"]
    out["residual_raw"] = out["y_raw"] - out["pred_raw"]
    return out


def prediction_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    y = predictions["y_scaled"].to_numpy(dtype=float)
    pred = predictions["pred_scaled"].to_numpy(dtype=float)
    residual = y - pred
    if len(predictions) > 1 and np.std(y) > 0 and np.std(pred) > 0:
        corr = float(np.corrcoef(y, pred)[0, 1])
    else:
        corr = float("nan")
    return {
        "n": int(len(predictions)),
        "residual_mean_scaled": float(np.mean(residual)),
        "residual_var_scaled": float(np.var(residual, ddof=1)) if len(residual) > 1 else float("nan"),
        "residual_var_raw": float(np.var(predictions["residual_raw"], ddof=1)) if len(residual) > 1 else float("nan"),
        "rmse_scaled": float(np.sqrt(np.mean(residual**2))),
        "rmse_raw": float(np.sqrt(np.mean(np.asarray(predictions["residual_raw"], dtype=float) ** 2))),
        "prediction_var_scaled": float(np.var(pred, ddof=1)) if len(pred) > 1 else float("nan"),
        "correlation": corr,
    }


def cell_dir(config: dict[str, Any], replication_id: int, s_train: int, loss_name: str | None = None) -> Path:
    base = Path(config["output_dir"])
    methods = configured_methods(config)
    if loss_name is None:
        loss_name = methods[0].name
    if len(methods) > 1:
        base = base / str(loss_name).lower()
    return base / f"rep_{int(replication_id):02d}" / f"s_{int(s_train):04d}"


def write_cell_manifest(
    output_dir: Path,
    config: dict[str, Any],
    replication_id: int,
    s_train: int,
    loss_name: str,
    training_loss: str,
    early_stopping_metric: str,
    bundle: SplitBundle,
) -> None:
    manifest = {
        "replication_id": int(replication_id),
        "s_train": int(s_train),
        "loss": str(loss_name).lower(),
        "method": str(loss_name).lower(),
        "training_loss": str(training_loss).lower(),
        "early_stopping_metric": str(early_stopping_metric).lower(),
        "counts": {
            "L": int(len(bundle.l_ids)),
            "V_stop": int(len(bundle.v_stop_ids)),
            "V_scale": int(len(bundle.v_scale_ids)),
            "L_prime": int(len(bundle.l_prime_ids)),
            "train": int(s_train),
            "E_eval": int(len(bundle.e_eval_ids)),
        },
        "hashes": {
            "L": ids_hash(bundle.l_ids),
            "V_stop": ids_hash(bundle.v_stop_ids),
            "V_scale": ids_hash(bundle.v_scale_ids),
            "L_prime": ids_hash(bundle.l_prime_ids),
            "train": ids_hash(train_ids_for_s(bundle, s_train)),
            "E_eval": ids_hash(bundle.e_eval_ids),
        },
        "seed": _rep_seed(config, replication_id),
    }
    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def cleanup_training_artifacts(output_dir: Path) -> None:
    for name in ("checkpoint", "checkpoints", "final_adapter"):
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)


def train_cell(config: dict[str, Any], replication_id: int, s_train: int, loss_name: str = "var") -> Path:
    method = method_by_name(config, loss_name)
    method_name = method.name
    training_loss = method.training_loss
    early_stopping_metric = method.early_stopping_metric
    output_dir = cell_dir(config, replication_id, s_train, method_name)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        print(f"skip_completed method={method_name} rep={replication_id} s={s_train} metrics={metrics_path}", flush=True)
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    clean = clean_wine(config["input_csv"])
    population_var_y_scaled = float(np.var(scaled_y(clean[Y_COL]), ddof=1))
    bundle = build_replication_splits(clean, config, replication_id)
    train_df = subset_by_ids(clean, train_ids_for_s(bundle, s_train), "train")
    stop_df = subset_by_ids(clean, bundle.v_stop_ids, "validation_stop")
    scale_df = subset_by_ids(clean, bundle.v_scale_ids, "validation_scale")
    eval_df = subset_by_ids(clean, bundle.e_eval_ids, "eval") if len(bundle.e_eval_ids) else None
    write_cell_manifest(output_dir, config, replication_id, s_train, method_name, training_loss, early_stopping_metric, bundle)

    requested_batch = int(config["batch_size"])
    batch_candidates = [requested_batch]
    fallback_values = config.get("oom_fallback_batch_sizes", None)
    if fallback_values is None:
        fallback_values = [config.get("oom_fallback_batch_size", 64)]
    if not isinstance(fallback_values, list):
        fallback_values = [fallback_values]
    for fallback_value in fallback_values:
        fallback_batch = int(fallback_value)
        if fallback_batch > 0 and fallback_batch not in batch_candidates:
            batch_candidates.append(fallback_batch)
    last_error: str | None = None
    used_oom_fallback = False
    for batch_index, batch_size in enumerate(batch_candidates):
        try:
            print(
                "train_cell_start",
                f"method={method_name}",
                f"loss={training_loss}",
                f"stop_metric={early_stopping_metric}",
                f"rep={replication_id}",
                f"s={s_train}",
                f"batch_size={batch_size}",
                f"max_epochs={int(config.get('max_epochs', 12))}",
                f"steps_per_epoch={steps_per_epoch_for_s(s_train, batch_size)}",
                flush=True,
            )
            runtime, stop_pred, scale_pred, eval_pred = train_once(
                config,
                train_df,
                stop_df,
                scale_df,
                eval_df,
                batch_size=batch_size,
                seed=_rep_seed(
                    config,
                    replication_id,
                    salt=0 if bool(config.get("common_train_seed_within_rep", True)) else int(s_train),
                ),
                loss_name=training_loss,
                early_stopping_metric=early_stopping_metric,
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
            print(f"cuda_oom_retry method={method_name} rep={replication_id} s={s_train} next_batch={next_batch}", flush=True)
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

    stop_pred["replication_id"] = int(replication_id)
    stop_pred["s_train"] = int(s_train)
    stop_pred["split_role"] = "validation_stop"
    scale_pred["replication_id"] = int(replication_id)
    scale_pred["s_train"] = int(s_train)
    scale_pred["split_role"] = "validation_scale"
    if bool(config.get("save_predictions", False)):
        stop_pred.to_parquet(output_dir / "validation_stop_predictions.parquet", index=False)
        scale_pred.to_parquet(output_dir / "validation_scale_predictions.parquet", index=False)
    epoch_history = runtime.pop("epoch_history", [])
    if epoch_history:
        pd.DataFrame(epoch_history).to_csv(output_dir / "epoch_history.csv", index=False)
    eval_metrics = None
    if eval_pred is not None:
        eval_pred["replication_id"] = int(replication_id)
        eval_pred["s_train"] = int(s_train)
        eval_pred["split_role"] = "eval"
        eval_pred.to_parquet(output_dir / "eval_predictions.parquet", index=False)
        eval_metrics = prediction_metrics(eval_pred)
    metrics = {
        "replication_id": int(replication_id),
        "s_train": int(s_train),
        "budget_B": int(config["budget_B"]),
        "validation_stop_size": validation_stop_size(config),
        "validation_scale_size": validation_scale_size(config),
        "eval_size": int(config["eval_size"]),
        "n_eff": n_effective_labeled(config),
        "model_name": config["model_name"],
        "loss": method_name,
        "method": method_name,
        "training_loss": training_loss,
        "early_stopping_metric": early_stopping_metric,
        "population_var_y_scaled": population_var_y_scaled,
        "population_var_y_raw": float(np.var(clean[Y_COL].astype(float), ddof=1)),
        "validation_stop": prediction_metrics(stop_pred),
        "validation_scale": prediction_metrics(scale_pred),
        "eval": eval_metrics,
        "runtime": runtime,
    }
    write_json(metrics_path, metrics)
    cleanup_training_artifacts(output_dir)
    print(f"train_cell_done method={method_name} loss={training_loss} stop_metric={early_stopping_metric} rep={replication_id} s={s_train} metrics={metrics_path}", flush=True)
    return output_dir


def _load_cell_metrics(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    missing = []
    for method in configured_methods(config):
        loss_name = method.name
        for rep in config["replication_ids"]:
            for s_train in config["s_grid"]:
                path = cell_dir(config, int(rep), int(s_train), loss_name) / "metrics.json"
                if not path.exists():
                    missing.append(
                        {
                            "loss": loss_name,
                            "replication_id": int(rep),
                            "s_train": int(s_train),
                            "path": str(path),
                        }
                    )
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                method_name = str(metrics.get("method", metrics.get("loss", loss_name))).lower()
                training_loss = str(metrics.get("training_loss", method.training_loss)).lower()
                early_stopping_metric = str(metrics.get("early_stopping_metric", method.early_stopping_metric)).lower()
                row = {
                    "loss": method_name,
                    "method": method_name,
                    "training_loss": training_loss,
                    "early_stopping_metric": early_stopping_metric,
                    "replication_id": int(metrics["replication_id"]),
                    "s_train": int(metrics["s_train"]),
                    "budget_B": int(metrics["budget_B"]),
                    "validation_stop_size": int(metrics.get("validation_stop_size", metrics.get("validation_size", 0))),
                    "validation_scale_size": int(metrics.get("validation_scale_size", 0)),
                    "eval_size": int(metrics["eval_size"]),
                    "n_eff": int(metrics["n_eff"]),
                    "population_var_y_scaled": float(metrics["population_var_y_scaled"]),
                    "validation_stop_residual_var": float(metrics["validation_stop"]["residual_var_scaled"]),
                    "validation_stop_residual_mean": float(metrics["validation_stop"]["residual_mean_scaled"]),
                    "validation_stop_rmse": float(metrics["validation_stop"]["rmse_scaled"]),
                    "validation_stop_corr": float(metrics["validation_stop"]["correlation"]),
                    "validation_scale_residual_var": float(metrics["validation_scale"]["residual_var_scaled"]),
                    "validation_scale_residual_mean": float(metrics["validation_scale"]["residual_mean_scaled"]),
                    "validation_scale_rmse": float(metrics["validation_scale"]["rmse_scaled"]),
                    "validation_scale_corr": float(metrics["validation_scale"]["correlation"]),
                    "actual_batch_size": int(metrics["runtime"]["actual_batch_size"]),
                    "requested_batch_size": int(metrics["runtime"]["requested_batch_size"]),
                    "oom_fallback_used": bool(metrics["runtime"]["oom_fallback_used"]),
                    "max_steps": int(metrics["runtime"]["max_steps"]),
                    "max_epochs": int(metrics["runtime"].get("max_epochs", 0)),
                    "epochs_trained": int(metrics["runtime"].get("epochs_trained", 0)),
                    "best_epoch": int(metrics["runtime"].get("best_epoch", 0)),
                    "early_stopped": bool(metrics["runtime"].get("early_stopped", False)),
                    "steps_per_epoch": int(metrics["runtime"].get("steps_per_epoch", 0)),
                    "total_train_steps": int(metrics["runtime"].get("total_train_steps", metrics["runtime"]["max_steps"])),
                    "runtime_early_stopping_metric": str(metrics["runtime"].get("early_stopping_metric", early_stopping_metric)).lower(),
                    "best_validation_stop_metric_value": float(metrics["runtime"].get("best_validation_stop_metric_value", np.nan)),
                    "best_validation_stop_residual_var": float(metrics["runtime"].get("best_validation_stop_residual_var", np.nan)),
                    "best_validation_stop_mse": float(metrics["runtime"].get("best_validation_stop_mse", np.nan)),
                    "runtime_seconds": float(metrics["runtime"]["runtime_seconds"]),
                    "device": str(metrics["runtime"]["device"]),
                    "final_train_loss": float(metrics["runtime"]["final_train_loss"]),
                    "mean_train_loss": float(metrics["runtime"]["mean_train_loss"]),
                }
                if metrics.get("eval") is not None:
                    row["eval_residual_var"] = float(metrics["eval"]["residual_var_scaled"])
                    row["eval_rmse"] = float(metrics["eval"]["rmse_scaled"])
                    row["eval_corr"] = float(metrics["eval"]["correlation"])
                rows.append(row)
    if missing:
        missing_text = "\n".join(
            f"loss={m['loss']} rep={m['replication_id']} s={m['s_train']} path={m['path']}"
            for m in missing
        )
        raise FileNotFoundError("missing metrics.json files:\n" + missing_text)
    return pd.DataFrame(rows).sort_values(["loss", "replication_id", "s_train"]).reset_index(drop=True)


def _power_law(s: np.ndarray, a: float, alpha: float, b: float) -> np.ndarray:
    return float(a) * np.asarray(s, dtype=float) ** (-float(alpha)) + float(b)


def fit_scaling_law(train_sizes: np.ndarray, residual_vars: np.ndarray, population_var_y: float) -> dict[str, float]:
    from scipy.optimize import least_squares, minimize_scalar

    x = np.asarray(train_sizes, dtype=float)
    y = np.asarray(residual_vars, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return {"a": np.nan, "alpha": np.nan, "b": np.nan, "sse": np.nan, "rmse": np.nan, "r2": np.nan}

    b_upper = max(float(population_var_y), 1e-9)
    lower = np.array([1e-12, 0.02, 0.0], dtype=float)
    upper = np.array([np.inf, 1.5, b_upper], dtype=float)
    starts = []
    for alpha0 in [0.05, 0.1, 0.2, 0.5, 0.9, 1.3]:
        for b0 in [0.0, min(float(np.min(y)) * 0.25, b_upper), min(float(np.min(y)) * 0.75, b_upper)]:
            a0 = max(float(np.max(y) - b0), 1e-6) * float(np.min(x) ** alpha0)
            starts.append(np.array([a0, alpha0, b0], dtype=float))

    best = None
    for start in starts:
        result = least_squares(
            lambda params: _power_law(x, params[0], params[1], params[2]) - y,
            x0=np.maximum(start, lower),
            bounds=(lower, upper),
            max_nfev=20_000,
        )
        sse = float(np.sum(result.fun**2))
        if best is None or sse < best["sse"]:
            best = {"params": result.x, "sse": sse, "success": bool(result.success)}
    if best is None:
        return {"a": np.nan, "alpha": np.nan, "b": np.nan, "sse": np.nan, "rmse": np.nan, "r2": np.nan}
    a, alpha, b = [float(v) for v in best["params"]]
    pred = _power_law(x, a, alpha, b)
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "a": a,
        "alpha": alpha,
        "b": b,
        "sse": float(best["sse"]),
        "rmse": float(np.sqrt(best["sse"] / len(x))),
        "r2": float(1.0 - best["sse"] / sst) if sst > 0 else float("nan"),
        "success": bool(best["success"]),
    }


def fitted_optimum(fit: dict[str, float], n_eff: int) -> dict[str, float]:
    from scipy.optimize import minimize_scalar

    params = np.array([fit.get("a", np.nan), fit.get("alpha", np.nan), fit.get("b", np.nan)], dtype=float)
    if not np.isfinite(params).all():
        return {"s_star": np.nan, "objective": np.nan}

    def objective(s_value: float) -> float:
        correction = float(n_eff) - float(s_value)
        if correction <= 0:
            return float("inf")
        return float(_power_law(np.array([s_value]), params[0], params[1], params[2])[0] / correction)

    result = minimize_scalar(objective, bounds=(1.0, float(n_eff - 1)), method="bounded")
    return {"s_star": float(result.x), "objective": float(result.fun)}


def discrete_objective(residual_var: float, s_train: int, n_eff: int) -> float:
    correction = int(n_eff) - int(s_train)
    if correction <= 0:
        return float("nan")
    return float(residual_var) / float(correction)


def build_scaling_fits(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    if "loss" not in metrics.columns:
        metrics["loss"] = "var"
    rows = []
    for (loss_name, rep), group in metrics.groupby(["loss", "replication_id"]):
        group = group.sort_values("s_train")
        n_eff = int(group["n_eff"].iloc[0])
        population_var = float(group["population_var_y_scaled"].iloc[0])
        sources = [("validation_scale", "validation_scale_residual_var")]
        if "eval_residual_var" in group.columns and group["eval_residual_var"].notna().any():
            sources.append(("eval", "eval_residual_var"))
        for source, col in sources:
            fit = fit_scaling_law(group["s_train"].to_numpy(), group[col].to_numpy(), population_var)
            optimum = fitted_optimum(fit, n_eff)
            observed_objectives = [
                discrete_objective(row[col], row["s_train"], n_eff) for _, row in group.iterrows()
            ]
            best_idx = int(np.nanargmin(observed_objectives))
            best_row = group.iloc[best_idx]
            rows.append(
                {
                    "replication_id": int(rep),
                    "loss": str(loss_name),
                    "source": source,
                    "a": fit["a"],
                    "alpha": fit["alpha"],
                    "b": fit["b"],
                    "sse": fit["sse"],
                    "rmse": fit["rmse"],
                    "r2": fit["r2"],
                    "fit_success": fit.get("success", False),
                    "fitted_s_star": optimum["s_star"],
                    "fitted_objective": optimum["objective"],
                    "empirical_oracle_s": int(best_row["s_train"]),
                    "empirical_oracle_objective": float(observed_objectives[best_idx]),
                    "population_var_y_scaled": population_var,
                    "n_eff": n_eff,
                }
            )
    return pd.DataFrame(rows).sort_values(["loss", "replication_id", "source"]).reset_index(drop=True)


def replay_rampup(metrics: pd.DataFrame, min_points_for_stop: int = 4) -> pd.DataFrame:
    metrics = metrics.copy()
    if "loss" not in metrics.columns:
        metrics["loss"] = "var"
    rows = []
    oracle_col = (
        "eval_residual_var"
        if "eval_residual_var" in metrics.columns and metrics["eval_residual_var"].notna().any()
        else "validation_scale_residual_var"
    )
    oracle_source = "eval" if oracle_col == "eval_residual_var" else "validation_scale"
    for (loss_name, rep), group in metrics.groupby(["loss", "replication_id"]):
        group = group.sort_values("s_train").reset_index(drop=True)
        n_eff = int(group["n_eff"].iloc[0])
        population_var = float(group["population_var_y_scaled"].iloc[0])
        oracle_objectives = {
            int(row["s_train"]): discrete_objective(row[oracle_col], row["s_train"], n_eff)
            for _, row in group.iterrows()
        }
        oracle_s = min(oracle_objectives, key=oracle_objectives.get)
        oracle_objective = float(oracle_objectives[oracle_s])
        stop_record = None
        for stage_idx in range(len(group)):
            observed = group.iloc[: stage_idx + 1]
            current_largest = int(observed["s_train"].max())
            if len(observed) < min_points_for_stop:
                continue
            fit = fit_scaling_law(
                observed["s_train"].to_numpy(),
                observed["validation_scale_residual_var"].to_numpy(),
                population_var,
            )
            optimum = fitted_optimum(fit, n_eff)
            fitted_s_star = float(optimum["s_star"])
            is_last = stage_idx == len(group) - 1
            if is_last or (np.isfinite(fitted_s_star) and fitted_s_star <= current_largest):
                observed_sizes = observed["s_train"].to_numpy(dtype=int)
                fitted_objectives = [
                    discrete_objective(float(_power_law(np.array([s]), fit["a"], fit["alpha"], fit["b"])[0]), int(s), n_eff)
                    for s in observed_sizes
                ]
                selected_idx = int(np.nanargmin(fitted_objectives))
                ramp_s = int(observed_sizes[selected_idx])
                ramp_oracle_source_objective = float(oracle_objectives[ramp_s])
                stop_record = {
                    "replication_id": int(rep),
                    "loss": str(loss_name),
                    "oracle_source": oracle_source,
                    "stopped_stage_index": int(stage_idx + 1),
                    "stopped_current_largest_s": int(current_largest),
                    "observed_train_sizes": ",".join(str(int(x)) for x in observed_sizes),
                    "ramp_fitted_s_star": fitted_s_star,
                    "ramp_s_best_seen": ramp_s,
                    "ramp_fitted_objective_best_seen": float(fitted_objectives[selected_idx]),
                    "oracle_s": int(oracle_s),
                    "ramp_oracle_source_objective": ramp_oracle_source_objective,
                    "oracle_source_objective": oracle_objective,
                    "regret": (ramp_oracle_source_objective - oracle_objective) / oracle_objective
                    if oracle_objective > 0
                    else float("nan"),
                    "exact_oracle_match": bool(ramp_s == oracle_s),
                    "fit_a": fit["a"],
                    "fit_alpha": fit["alpha"],
                    "fit_b": fit["b"],
                    "fit_r2": fit["r2"],
                    "n_eff": n_eff,
                }
                break
        if stop_record is None:
            raise RuntimeError(f"ramp-up replay did not stop for replication {rep}")
        rows.append(stop_record)
    return pd.DataFrame(rows).sort_values(["loss", "replication_id"]).reset_index(drop=True)


def summarize_rampup(ramp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = ["loss"] if "loss" in ramp.columns else [lambda _: "all"]
    for key, group in ramp.groupby(group_keys, dropna=False):
        regrets = pd.to_numeric(group["regret"], errors="coerce")
        row = {
            "n_replications": int(len(group)),
            "mean_regret": float(regrets.mean()),
            "median_regret": float(regrets.median()),
            "sd_regret": float(regrets.std(ddof=1)) if len(regrets) > 1 else 0.0,
            "min_regret": float(regrets.min()),
            "max_regret": float(regrets.max()),
            "exact_oracle_match_rate": float(group["exact_oracle_match"].mean()),
            "mean_ramp_s_best_seen": float(group["ramp_s_best_seen"].mean()),
            "mean_oracle_s": float(group["oracle_s"].mean()),
        }
        if "loss" in ramp.columns:
            row["loss"] = str(key[0] if isinstance(key, tuple) else key)
        rows.append(row)
    cols = ["loss"] + [c for c in rows[0] if c != "loss"] if rows and "loss" in rows[0] else None
    return pd.DataFrame(rows)[cols] if cols else pd.DataFrame(rows)


def build_loss_comparison_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    if "loss" not in metrics.columns:
        metrics["loss"] = "var"
    value_cols = [
        "validation_scale_residual_var",
        "validation_scale_rmse",
        "validation_scale_corr",
        "eval_residual_var",
        "eval_rmse",
        "eval_corr",
        "epochs_trained",
        "best_epoch",
        "runtime_seconds",
    ]
    present = [col for col in value_cols if col in metrics.columns]
    summary = (
        metrics.groupby(["loss", "s_train"], as_index=False)[present]
        .mean(numeric_only=True)
        .sort_values(["loss", "s_train"])
        .reset_index(drop=True)
    )
    if set(metrics["loss"].unique()) >= {"mse", "var"}:
        wide = summary.pivot(index="s_train", columns="loss", values="validation_scale_residual_var")
        if {"mse", "var"}.issubset(wide.columns):
            deltas = pd.DataFrame(
                {
                    "loss": "mse_minus_var",
                    "s_train": wide.index.astype(int),
                    "validation_scale_residual_var": (wide["mse"] - wide["var"]).to_numpy(),
                    "validation_scale_relative_delta": ((wide["mse"] - wide["var"]) / wide["var"]).to_numpy(),
                }
            )
            summary = pd.concat([summary, deltas], ignore_index=True, sort=False)
    named_methods = set(metrics["loss"].unique())
    comparison_pairs = [
        ("mse_stop_mse", "var_stop_var"),
        ("mse_stop_var", "var_stop_var"),
        ("mse_stop_mse", "mse_stop_var"),
    ]
    delta_rows = []
    for left, right in comparison_pairs:
        if {left, right}.issubset(named_methods):
            for s_train in sorted(metrics["s_train"].unique()):
                row = {
                    "loss": f"{left}_minus_{right}",
                    "s_train": int(s_train),
                }
                left_rows = metrics[(metrics["loss"] == left) & (metrics["s_train"] == s_train)]
                right_rows = metrics[(metrics["loss"] == right) & (metrics["s_train"] == s_train)]
                for col in ["validation_scale_residual_var", "eval_residual_var"]:
                    if col not in metrics.columns:
                        continue
                    left_value = float(left_rows[col].mean())
                    right_value = float(right_rows[col].mean())
                    row[col] = left_value - right_value
                    row[f"{col}_relative_delta"] = (
                        (left_value - right_value) / right_value if right_value != 0 else float("nan")
                    )
                delta_rows.append(row)
    if delta_rows:
        summary = pd.concat([summary, pd.DataFrame(delta_rows)], ignore_index=True, sort=False)
    return summary


def leakage_check(clean: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    failures = []
    rep_reports = []
    s_grid = [int(x) for x in config["s_grid"]]
    for rep in config["replication_ids"]:
        try:
            bundle = build_replication_splits(clean, config, int(rep))
            validate_split_bundle(bundle, s_grid)
            rep_reports.append(
                {
                    "replication_id": int(rep),
                    "counts": {
                        "L": len(bundle.l_ids),
                        "V_stop": len(bundle.v_stop_ids),
                        "V_scale": len(bundle.v_scale_ids),
                        "L_prime": len(bundle.l_prime_ids),
                        "E_eval": len(bundle.e_eval_ids),
                    },
                    "hashes": {
                        "L": ids_hash(bundle.l_ids),
                        "V_stop": ids_hash(bundle.v_stop_ids),
                        "V_scale": ids_hash(bundle.v_scale_ids),
                        "L_prime": ids_hash(bundle.l_prime_ids),
                        "E_eval": ids_hash(bundle.e_eval_ids),
                    },
                }
            )
        except Exception as exc:
            failures.append({"replication_id": int(rep), "error": str(exc)})
    return {
        "passed": not failures,
        "failures": failures,
        "replications": rep_reports,
        "eval_usage": (
            "E_eval is disabled for this timing run."
            if int(config.get("eval_size", 0)) == 0
            else "E_eval is sampled from P minus L and is used only by aggregate ex-post diagnostics."
        ),
    }


def write_figures(metrics: pd.DataFrame, scaling: pd.DataFrame, ramp: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    metrics = metrics.copy()
    scaling = scaling.copy()
    ramp = ramp.copy()
    if "loss" not in metrics.columns:
        metrics["loss"] = "var"
    if "loss" not in scaling.columns:
        scaling["loss"] = "var"
    if "loss" not in ramp.columns:
        ramp["loss"] = "var"

    s_grid = np.sort(metrics["s_train"].unique())
    with PdfPages(output_dir / "scaling_law_full_grid.pdf") as pdf:
        sources = [("validation_scale", "validation_scale_residual_var")]
        if "eval_residual_var" in metrics.columns and metrics["eval_residual_var"].notna().any():
            sources.append(("eval", "eval_residual_var"))
        for loss_name in sorted(metrics["loss"].unique()):
            loss_metrics = metrics[metrics["loss"] == loss_name]
            for source, col in sources:
                if col not in loss_metrics.columns or loss_metrics[col].isna().all():
                    continue
                fig, ax = plt.subplots(figsize=(8, 5))
                for rep, group in loss_metrics.groupby("replication_id"):
                    group = group.sort_values("s_train")
                    ax.plot(group["s_train"], group[col], marker="o", alpha=0.35, label=f"rep {rep}" if rep == 0 else None)
                    fit_row = scaling[
                        (scaling["loss"] == loss_name)
                        & (scaling["replication_id"] == rep)
                        & (scaling["source"] == source)
                    ].iloc[0]
                    dense = np.linspace(float(s_grid.min()), float(s_grid.max()), 200)
                    ax.plot(dense, _power_law(dense, fit_row["a"], fit_row["alpha"], fit_row["b"]), alpha=0.25)
                ax.set_title(f"{loss_name} {source} residual variance scaling law")
                ax.set_xlabel("training size s")
                ax.set_ylabel("scaled residual variance")
                ax.grid(True, alpha=0.25)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    comparison_sources = [("validation_scale", "validation_scale_residual_var")]
    if "eval_residual_var" in metrics.columns and metrics["eval_residual_var"].notna().any():
        comparison_sources.append(("eval", "eval_residual_var"))
    with PdfPages(output_dir / "loss_comparison_residual_variance.pdf") as pdf:
        for source, col in comparison_sources:
            if col not in metrics.columns or metrics[col].isna().all():
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for loss_name, loss_group in metrics.groupby("loss"):
                loss_group = loss_group.sort_values("s_train")
                for _, rep_group in loss_group.groupby("replication_id"):
                    rep_group = rep_group.sort_values("s_train")
                    ax.plot(rep_group["s_train"], rep_group[col], alpha=0.18)
                mean_curve = loss_group.groupby("s_train", as_index=False)[col].mean()
                ax.plot(mean_curve["s_train"], mean_curve[col], marker="o", linewidth=2.5, label=f"{loss_name} mean")
            ax.set_title(f"{source} residual variance by loss")
            ax.set_xlabel("training size s")
            ax.set_ylabel("scaled residual variance")
            ax.grid(True, alpha=0.25)
            ax.legend()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    with PdfPages(output_dir / "rampup_stagewise_fits.pdf") as pdf:
        for _, ramp_row in ramp.iterrows():
            rep = int(ramp_row["replication_id"])
            loss_name = str(ramp_row["loss"])
            group = metrics[(metrics["loss"] == loss_name) & (metrics["replication_id"] == rep)].sort_values("s_train")
            observed_sizes = [int(x) for x in str(ramp_row["observed_train_sizes"]).split(",") if x]
            observed = group[group["s_train"].isin(observed_sizes)]
            dense = np.linspace(float(group["s_train"].min()), float(group["s_train"].max()), 200)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(group["s_train"], group["validation_scale_residual_var"], marker="o", label="V_scale full grid")
            ax.plot(
                dense,
                _power_law(dense, ramp_row["fit_a"], ramp_row["fit_alpha"], ramp_row["fit_b"]),
                label="stagewise fit at stop",
            )
            ax.scatter(
                observed["s_train"],
                observed["validation_scale_residual_var"],
                s=80,
                facecolors="none",
                edgecolors="black",
                label="observed before stop",
            )
            ax.axvline(ramp_row["ramp_s_best_seen"], color="tab:green", linestyle="--", label="ramp selected")
            ax.axvline(
                ramp_row["oracle_s"],
                color="tab:red",
                linestyle=":",
                label=f"{ramp_row['oracle_source']} oracle",
            )
            ax.set_title(f"{loss_name} rep {rep} ramp-up replay")
            ax.set_xlabel("training size s")
            ax.set_ylabel("V_scale residual variance")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ramp["regret"], bins=min(10, max(3, len(ramp))), edgecolor="black")
    ax.set_title("Ramp-up regret distribution")
    ax.set_xlabel("relative regret")
    ax.set_ylabel("replications")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_dir / "rampup_regret_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate(config: dict[str, Any]) -> Path:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = clean_wine(config["input_csv"])
    leak_report = leakage_check(clean, config)
    write_json(output_dir / "leakage_check_report.json", leak_report)
    if not leak_report["passed"]:
        raise RuntimeError("leakage check failed")

    metrics = _load_cell_metrics(config)
    metrics.to_csv(output_dir / "training_runtime_summary.csv", index=False)
    scaling = build_scaling_fits(metrics)
    scaling.to_csv(output_dir / "scaling_fit_by_rep.csv", index=False)
    ramp = replay_rampup(metrics, min_points_for_stop=int(config["min_points_for_stop"]))
    ramp.to_csv(output_dir / "rampup_recovery_by_rep.csv", index=False)
    ramp_summary = summarize_rampup(ramp)
    ramp_summary.to_csv(output_dir / "rampup_recovery_summary.csv", index=False)
    loss_summary = build_loss_comparison_summary(metrics)
    loss_summary.to_csv(output_dir / "loss_comparison_summary.csv", index=False)
    write_figures(metrics, scaling, ramp, output_dir)
    print(f"aggregate_done output_dir={output_dir}", flush=True)
    print(ramp_summary.to_string(index=False), flush=True)
    return output_dir


def task_index_to_rep_s(config: dict[str, Any], task_index: int) -> tuple[int, int]:
    s_grid = [int(x) for x in config["s_grid"]]
    reps = [int(x) for x in config["replication_ids"]]
    total = len(s_grid) * len(reps)
    if task_index < 0 or task_index >= total:
        raise ValueError(f"task_index must be in [0, {total - 1}]")
    rep = reps[task_index // len(s_grid)]
    s_train = s_grid[task_index % len(s_grid)]
    return rep, s_train


def task_index_to_loss_rep_s(config: dict[str, Any], task_index: int) -> tuple[str, int, int]:
    methods = configured_methods(config)
    s_grid = [int(x) for x in config["s_grid"]]
    reps = [int(x) for x in config["replication_ids"]]
    cells_per_loss = len(s_grid) * len(reps)
    total = len(methods) * cells_per_loss
    if task_index < 0 or task_index >= total:
        raise ValueError(f"task_index must be in [0, {total - 1}]")
    loss_name = methods[task_index // cells_per_loss].name
    within_loss = task_index % cells_per_loss
    rep = reps[within_loss // len(s_grid)]
    s_train = s_grid[within_loss % len(s_grid)]
    return loss_name, rep, s_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wine Reviews LoRA-Var scaling-law experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train-cell")
    train_parser.add_argument("--config", required=True)
    train_group = train_parser.add_mutually_exclusive_group(required=True)
    train_group.add_argument("--task-index", type=int)
    train_group.add_argument("--rep-s", nargs=2, type=int, metavar=("REP", "S"))
    train_parser.add_argument("--loss")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "train-cell":
        if args.task_index is not None:
            loss_name, rep, s_train = task_index_to_loss_rep_s(config, args.task_index)
        else:
            rep, s_train = int(args.rep_s[0]), int(args.rep_s[1])
            loss_name = args.loss or configured_methods(config)[0].name
        train_cell(config, rep, s_train, loss_name=loss_name)
    elif args.command == "aggregate":
        aggregate(config)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
