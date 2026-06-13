from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.build_wine import build_artifacts
from src.eval.predict import save_prediction_frame
from src.models.lora_regression import LoraRegressionConfig, format_wine_text, load_lora_regression_model, load_tokenizer


class VarianceRegressionTrainerMixin:
    """Trainer mixin that replaces MSE with residual batch variance."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # pragma: no cover - needs torch
        labels = inputs.pop("labels").reshape(-1)
        outputs = model(**inputs)
        pred = outputs.logits.reshape(-1)
        residual = labels - pred
        loss = ((residual - residual.mean()) ** 2).mean()
        return (loss, outputs) if return_outputs else loss


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _require_training_modules():
    try:
        import torch
        from datasets import Dataset
        from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError(
            "Training requires torch, datasets, transformers, peft, accelerate, and optionally bitsandbytes."
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def _training_args(cls, kwargs: dict[str, Any]):
    supported = set(inspect.signature(cls.__init__).parameters)
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    if "eval_strategy" in supported and "evaluation_strategy" in kwargs:
        filtered["eval_strategy"] = kwargs["evaluation_strategy"]
    return cls(**filtered)


def _merge_population_roles(population: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    df = population.merge(roles[["sample_id", "split_role"]], on="sample_id", how="left", validate="one_to_one")
    if df["split_role"].isna().any():
        raise ValueError("some population rows are missing split roles")
    df["text"] = df["description"].map(format_wine_text)
    return df


def _standardize_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    train_points = df.loc[df["split_role"] == "train", "points"].astype(float)
    mean = float(train_points.mean())
    std = float(train_points.std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    out = df.copy()
    out["labels"] = (out["points"].astype(float) - mean) / std
    return out, mean, std


def _tokenize_dataset(df: pd.DataFrame, tokenizer, max_length: int, Dataset):
    dataset = Dataset.from_pandas(df[["sample_id", "text", "labels"]], preserve_index=False)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tok, batched=True)
    return tokenized.remove_columns(["sample_id", "text"])


def run_training(config: dict[str, Any], loss: str) -> Path:
    if loss not in {"mse", "var"}:
        raise ValueError("loss must be one of: mse, var")
    mods = _require_training_modules()

    run_dir = Path(config["output_dir"]) / loss
    run_dir.mkdir(parents=True, exist_ok=True)

    population, roles = build_artifacts(
        input_csv=config.get("input_csv", "Code/wine_data.csv"),
        output_dir=run_dir,
        population_size=config.get("population_size"),
        budget=int(config["budget"]),
        train_size=int(config["train_size"]),
        validation_size=int(config["validation_size"]),
        seed=int(config.get("seed", 20260612)),
        replication_id=int(config.get("replication_id", 0)),
        population_seed=int(config["population_seed"]) if "population_seed" in config else None,
        split_seed=int(config["split_seed"]) if "split_seed" in config else None,
    )
    df = _merge_population_roles(population, roles)
    df, label_mean, label_std = _standardize_labels(df)
    with open(run_dir / "label_stats.json", "w", encoding="utf-8") as f:
        json.dump({"label_mean": label_mean, "label_std": label_std}, f, indent=2)

    model_name = config.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    max_length = int(config.get("max_length", 256))
    tokenizer = load_tokenizer(model_name, max_length)
    lora_cfg = LoraRegressionConfig(
        model_name=model_name,
        fallback_model_name=config.get("fallback_model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
        max_length=max_length,
        load_in_4bit=bool(config.get("load_in_4bit", True)),
        bf16=bool(config.get("bf16", True)),
        fp16=bool(config.get("fp16", False)),
        device_map=config.get("device_map", "auto"),
        lora_r=int(config.get("lora_r", 4)),
        lora_alpha=int(config.get("lora_alpha", 4)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=tuple(config.get("target_modules", ["q_proj", "v_proj"])),
    )
    model, resolved_model_name = load_lora_regression_model(lora_cfg)
    tokenizer = load_tokenizer(resolved_model_name, max_length)

    train_df = df[df["split_role"] == "train"].reset_index(drop=True)
    validation_df = df[df["split_role"] == "validation"].reset_index(drop=True)
    train_ds = _tokenize_dataset(train_df, tokenizer, max_length, mods["Dataset"])
    val_ds = _tokenize_dataset(validation_df, tokenizer, max_length, mods["Dataset"])
    all_ds = _tokenize_dataset(df.reset_index(drop=True), tokenizer, max_length, mods["Dataset"])

    collator = mods["DataCollatorWithPadding"](tokenizer=tokenizer)
    args = _training_args(
        mods["TrainingArguments"],
        {
            "output_dir": str(run_dir / "checkpoints"),
            "learning_rate": float(config.get("learning_rate", 2e-5)),
            "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 2)),
            "per_device_eval_batch_size": int(config.get("per_device_eval_batch_size", 8)),
            "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 1)),
            "max_steps": int(config.get("max_steps", 20)),
            "num_train_epochs": float(config.get("num_train_epochs", 1)),
            "weight_decay": float(config.get("weight_decay", 0.0)),
            "evaluation_strategy": config.get("evaluation_strategy", "steps"),
            "eval_steps": int(config.get("eval_steps", 10)),
            "save_strategy": config.get("save_strategy", "no"),
            "logging_steps": int(config.get("logging_steps", 5)),
            "bf16": bool(config.get("bf16", True)),
            "fp16": bool(config.get("fp16", False)),
            "report_to": [],
            "remove_unused_columns": True,
        },
    )

    base_trainer = mods["Trainer"]
    trainer_cls = type("VarianceRegressionTrainer", (VarianceRegressionTrainerMixin, base_trainer), {}) if loss == "var" else base_trainer
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    if bool(config.get("save_adapter", True)):
        adapter_dir = run_dir / "final_adapter"
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    pred = trainer.predict(all_ds).predictions.reshape(-1)
    save_prediction_frame(
        output_path=run_dir / "predictions.parquet",
        population=df,
        pred_scaled=pred,
        label_mean=label_mean,
        label_std=label_std,
        method=f"lora_{loss}",
        model_name=resolved_model_name,
        loss=loss,
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA continuous-regression Wine model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--loss", choices=["mse", "var"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_training(load_config(args.config), args.loss)
    print(f"saved run to {run_dir}")


if __name__ == "__main__":
    main()
