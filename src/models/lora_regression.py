from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_MODEL_CANDIDATES = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]


@dataclass
class LoraRegressionConfig:
    model_name: str = DEFAULT_MODEL_CANDIDATES[0]
    fallback_model_name: str = DEFAULT_MODEL_CANDIDATES[1]
    max_length: int = 256
    load_in_4bit: bool = True
    bf16: bool = True
    fp16: bool = False
    device_map: str | dict[str, Any] | None = "auto"
    lora_r: int = 4
    lora_alpha: int = 4
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    zero_init_regression_head: bool = True


def format_wine_text(description: str) -> str:
    return (
        "You are an expert wine critic. Given a wine review, predict the expert "
        "rating on an 80-100 scale.\n\n"
        f"Review:\n{description}\n\n"
        "Rating:"
    )


def _require_hf_modules():
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "LoRA training requires torch, transformers, peft, accelerate, and optionally bitsandbytes. "
            "Install these on the Hyak environment before running training."
        ) from exc
    return {
        "torch": torch,
        "AutoTokenizer": AutoTokenizer,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
    }


def load_tokenizer(model_name: str, max_length: int):
    mods = _require_hf_modules()
    tokenizer = mods["AutoTokenizer"].from_pretrained(model_name, model_max_length=max_length)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _zero_init_regression_head(model, torch) -> None:
    for name in ("score", "classifier"):
        head = getattr(model, name, None)
        if head is None:
            continue
        weight = getattr(head, "weight", None)
        if weight is not None:
            torch.nn.init.zeros_(weight)
        bias = getattr(head, "bias", None)
        if bias is not None:
            torch.nn.init.zeros_(bias)


def load_lora_regression_model(config: LoraRegressionConfig):
    mods = _require_hf_modules()
    torch = mods["torch"]

    dtype = torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else None)
    quantization_config = None
    if config.load_in_4bit:
        quantization_config = mods["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype or torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    last_error: Exception | None = None
    for model_name in [config.model_name, config.fallback_model_name]:
        try:
            kwargs: dict[str, Any] = {
                "num_labels": 1,
                "device_map": config.device_map,
            }
            if dtype is not None:
                kwargs["torch_dtype"] = dtype
            if quantization_config is not None:
                kwargs["quantization_config"] = quantization_config
            model = mods["AutoModelForSequenceClassification"].from_pretrained(model_name, **kwargs)
            model.config.problem_type = "regression"
            if model.config.pad_token_id is None and getattr(model.config, "eos_token_id", None) is not None:
                model.config.pad_token_id = model.config.eos_token_id
            if config.zero_init_regression_head:
                _zero_init_regression_head(model, torch)
            if config.load_in_4bit:
                model = mods["prepare_model_for_kbit_training"](model)

            modules_to_save = [name for name in ("score", "classifier") if hasattr(model, name)]
            lora_config = mods["LoraConfig"](
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=list(config.target_modules),
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type=mods["TaskType"].SEQ_CLS,
                modules_to_save=modules_to_save or None,
            )
            model = mods["get_peft_model"](model, lora_config)
            model.print_trainable_parameters()
            return model, model_name
        except Exception as exc:  # pragma: no cover - depends on remote model/GPU environment
            last_error = exc
            continue
    raise RuntimeError(f"Could not load model candidates: {last_error}") from last_error
