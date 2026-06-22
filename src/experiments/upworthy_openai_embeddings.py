from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.experiments.upworthy_feature_screening import build_candidate_features


DEFAULT_INPUT_CSV = Path("Data/upworthy_pairs_with_text_features.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/upworthy_m_estimation/openai_embeddings")
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 512


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = " ".join(text.split())
    return text


def choose_screening_sample(
    frame: pd.DataFrame,
    *,
    n_pairs: int,
    n_eval: int,
    seed: int,
) -> pd.DataFrame:
    if n_eval <= 0 or n_eval >= n_pairs:
        raise ValueError("n_eval must be positive and smaller than n_pairs")
    n_train_pool = n_pairs - n_eval
    h_scale = frame.loc[frame["split"] == "h_scale"].copy()
    target = frame.loc[frame["split"] == "target"].copy()
    if len(h_scale) < n_train_pool:
        raise ValueError(f"need {n_train_pool} h_scale rows, found {len(h_scale)}")
    if len(target) < n_eval:
        raise ValueError(f"need {n_eval} target rows, found {len(target)}")
    train_pool = h_scale.sample(n=n_train_pool, random_state=seed).copy()
    evaluation = target.sample(n=n_eval, random_state=seed + 1).copy()
    train_pool["screen_split"] = "train_pool"
    evaluation["screen_split"] = "evaluation"
    out = pd.concat([train_pool, evaluation], ignore_index=True)
    out.insert(0, "screen_row_id", np.arange(len(out), dtype=int))
    return out


def ordered_unique_texts(frame: pd.DataFrame) -> list[str]:
    seen: set[str] = set()
    texts: list[str] = []
    for column in ["headline_a", "headline_b"]:
        for value in frame[column].tolist():
            text = normalize_text(value)
            if text and text not in seen:
                seen.add(text)
                texts.append(text)
    return texts


def call_embeddings_api(
    texts: list[str],
    *,
    api_key: str,
    model: str,
    dimensions: int | None,
    timeout_seconds: int,
    max_retries: int,
) -> np.ndarray:
    payload: dict[str, object] = {"model": model, "input": texts}
    if dimensions is not None:
        payload["dimensions"] = int(dimensions)
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                parsed = json.loads(response.read().decode("utf-8"))
            rows = sorted(parsed["data"], key=lambda item: item["index"])
            return np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenAI embeddings HTTP {exc.code}: {message[:500]}")
        except Exception as exc:  # pragma: no cover - network failures vary.
            last_error = exc
        if attempt < max_retries:
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"embedding request failed after {max_retries + 1} attempts: {last_error}")


def embed_texts(
    texts: list[str],
    *,
    api_key: str,
    model: str,
    dimensions: int | None,
    batch_size: int,
    timeout_seconds: int,
    max_retries: int,
) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        matrix = call_embeddings_api(
            batch,
            api_key=api_key,
            model=model,
            dimensions=dimensions,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        for text, embedding in zip(batch, matrix):
            embeddings[text] = embedding
        print(f"embedded {min(start + len(batch), total)}/{total}", flush=True)
    return embeddings


def matrix_for_column(frame: pd.DataFrame, column: str, embeddings: dict[str, np.ndarray]) -> np.ndarray:
    rows = []
    missing = 0
    dim = len(next(iter(embeddings.values())))
    zero = np.zeros(dim, dtype=np.float32)
    for value in frame[column].tolist():
        text = normalize_text(value)
        embedding = embeddings.get(text)
        if embedding is None:
            missing += 1
            embedding = zero
        rows.append(embedding)
    if missing:
        print(f"warning: {missing} missing embeddings for {column}; filled with zeros", flush=True)
    return np.vstack(rows).astype(np.float32)


def save_embedding_cache(
    frame: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    *,
    output_npz: Path,
    dtype: str,
) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    embedding_a = matrix_for_column(frame, "headline_a", embeddings)
    embedding_b = matrix_for_column(frame, "headline_b", embeddings)
    if dtype == "float16":
        embedding_a = embedding_a.astype(np.float16)
        embedding_b = embedding_b.astype(np.float16)
    elif dtype == "float32":
        embedding_a = embedding_a.astype(np.float32)
        embedding_b = embedding_b.astype(np.float32)
    else:
        raise ValueError("dtype must be float16 or float32")
    np.savez_compressed(
        output_npz,
        embedding_a=embedding_a,
        embedding_b=embedding_b,
        screen_row_id=frame["screen_row_id"].to_numpy(dtype=np.int64),
        pair_id=frame["pair_id"].astype(str).to_numpy(),
        screen_split=frame["screen_split"].astype(str).to_numpy(),
        y=frame["y_logit_ctr_diff"].astype(float).to_numpy(dtype=np.float32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache OpenAI embeddings for a small Upworthy pair pilot.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--n-pairs", type=int, default=5000)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build sample files and metadata without calling the API.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input_csv)
    feature_frame, metadata = build_candidate_features(frame)
    sample = choose_screening_sample(feature_frame, n_pairs=args.n_pairs, n_eval=args.n_eval, seed=args.seed)

    stem = f"upworthy_openai_{args.model}_d{args.dimensions}_n{args.n_pairs}_seed{args.seed}"
    output_csv = args.output_dir / f"{stem}_sample.csv"
    output_npz = args.output_dir / f"{stem}_{args.dtype}.npz"
    output_meta = args.output_dir / f"{stem}_metadata.json"
    output_feature_meta = args.output_dir / f"{stem}_feature_metadata.csv"

    if output_npz.exists() and not args.force and not args.dry_run:
        raise FileExistsError(f"{output_npz} already exists; pass --force to overwrite")

    sample.to_csv(output_csv, index=False)
    metadata.to_csv(output_feature_meta, index=False)
    texts = ordered_unique_texts(sample)
    payload = {
        "input_csv": str(args.input_csv),
        "sample_csv": str(output_csv),
        "feature_metadata_csv": str(output_feature_meta),
        "embedding_npz": str(output_npz),
        "model": args.model,
        "dimensions": int(args.dimensions),
        "n_pairs": int(len(sample)),
        "n_train_pool": int((sample["screen_split"] == "train_pool").sum()),
        "n_evaluation": int((sample["screen_split"] == "evaluation").sum()),
        "n_unique_texts": int(len(texts)),
        "dtype": args.dtype,
        "seed": int(args.seed),
        "dry_run": bool(args.dry_run),
    }
    write_json(output_meta, payload)

    if args.dry_run:
        print(json.dumps(payload, indent=2), flush=True)
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Set it in the shell environment; do not write it to files.")
    embeddings = embed_texts(
        texts,
        api_key=api_key,
        model=args.model,
        dimensions=args.dimensions,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    save_embedding_cache(sample, embeddings, output_npz=output_npz, dtype=args.dtype)
    payload["embedding_shape"] = list(matrix_for_column(sample.head(1), "headline_a", embeddings).shape)
    write_json(output_meta, payload)
    print(f"wrote {output_npz}", flush=True)


if __name__ == "__main__":
    main()
