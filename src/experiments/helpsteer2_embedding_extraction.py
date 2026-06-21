from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_pair_texts(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    required = {"text_1", "text_2"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing pair text columns: {missing}")
    return frame["text_1"].astype(str).tolist(), frame["text_2"].astype(str).tolist()


def pairwise_difference(response_1_embeddings: np.ndarray, response_2_embeddings: np.ndarray) -> np.ndarray:
    if response_1_embeddings.shape != response_2_embeddings.shape:
        raise ValueError(
            f"embedding shape mismatch: {response_1_embeddings.shape} vs {response_2_embeddings.shape}"
        )
    return response_2_embeddings - response_1_embeddings


def _last_nonpad_pool(last_hidden_state, attention_mask):
    import torch

    lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[batch_idx, lengths]


def _load_model_and_tokenizer(
    model_name: str,
    *,
    load_in_4bit: bool,
    dtype: str,
    trust_remote_code: bool,
):
    import torch
    from transformers import AutoModel, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs = {
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype_map[dtype]

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

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
    model = AutoModel.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def extract_text_embeddings(
    texts: Iterable[str],
    *,
    model_name: str,
    batch_size: int,
    max_length: int,
    load_in_4bit: bool,
    dtype: str,
    trust_remote_code: bool,
) -> np.ndarray:
    import torch

    tokenizer, model = _load_model_and_tokenizer(
        model_name,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    values = list(texts)
    outputs: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            result = model(**encoded)
            pooled = _last_nonpad_pool(result.last_hidden_state, encoded["attention_mask"])
        outputs.append(pooled.detach().float().cpu().numpy())
        print(f"embedded {min(start + batch_size, len(values))}/{len(values)}", flush=True)
    return np.vstack(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen pair embeddings for HelpSteer2 preference pairs.")
    parser.add_argument("--input-csv", default="Data/helpsteer2_preference_pairs.csv")
    parser.add_argument(
        "--output-npz",
        default="artifacts/helpsteer2_preference/embeddings/qwen_pair_embeddings.npz",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantized loading.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--save-response-embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    if args.max_rows is not None:
        frame = frame.head(args.max_rows).copy()
    text_1, text_2 = build_pair_texts(frame)

    print(
        {
            "rows": len(frame),
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "load_in_4bit": not args.no_4bit,
        },
        flush=True,
    )
    emb_1 = extract_text_embeddings(
        text_1,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        load_in_4bit=not args.no_4bit,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    emb_2 = extract_text_embeddings(
        text_2,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        load_in_4bit=not args.no_4bit,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    pair = pairwise_difference(emb_1, emb_2)

    output = Path(args.output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pair_embedding": pair.astype(np.float32),
        "sample_id": frame["sample_id"].to_numpy() if "sample_id" in frame.columns else np.arange(len(frame)),
        "y": frame["y_preference_strength"].to_numpy(dtype=np.float32),
        "feature_columns": np.array(["delta_log_length", "delta_prompt_coverage", "delta_format"]),
        "features": frame[["delta_log_length", "delta_prompt_coverage", "delta_format"]].to_numpy(dtype=np.float32),
    }
    if args.save_response_embeddings:
        payload["response_1_embedding"] = emb_1.astype(np.float32)
        payload["response_2_embedding"] = emb_2.astype(np.float32)
    np.savez_compressed(output, **payload)

    metadata = {
        "input_csv": args.input_csv,
        "output_npz": str(output),
        "model_name": args.model_name,
        "rows": int(len(frame)),
        "embedding_dim": int(pair.shape[1]),
        "pair_embedding_shape": list(pair.shape),
        "sign_convention": "pair_embedding = embedding(prompt,response_2) - embedding(prompt,response_1)",
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "load_in_4bit": not args.no_4bit,
        "dtype": args.dtype,
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
