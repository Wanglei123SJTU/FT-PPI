from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import re
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


PROMPT_COL = "prompt"
RESPONSE_1_COL = "response_1"
RESPONSE_2_COL = "response_2"
Y_COL = "preference_strength"
SPLIT_COL = "split"

DEFAULT_OUTPUT = Path("Data/helpsteer2_preference_pairs.csv")
HELPSTEER2_PREFERENCE_URL = (
    "https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/main/preference/preference.jsonl.gz"
)


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", str(text).lower())
    if keep_stopwords:
        return tokens
    return [token for token in tokens if token not in ENGLISH_STOP_WORDS]


def token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", str(text)))


def sentence_count(text: str) -> int:
    count = len(re.findall(r"[.!?]+", str(text)))
    return max(1, count)


def has_structured_format(text: str) -> int:
    value = str(text)
    patterns = [
        r"(?m)^\s*[-*+]\s+\S",
        r"(?m)^\s*\d+[\.)]\s+\S",
        r"(?m)^#{1,6}\s+\S",
        r"\*\*[^*]+\*\*",
        r"(?m)^\s*[A-Z][A-Za-z0-9 /-]{2,}:\s*$",
    ]
    return int(any(re.search(pattern, value) for pattern in patterns))


def prompt_coverage(prompt: str, response: str, *, keep_stopwords: bool = False) -> float:
    prompt_tokens = set(tokenize(prompt, keep_stopwords=keep_stopwords))
    if not prompt_tokens:
        return 0.0
    response_tokens = set(tokenize(response, keep_stopwords=keep_stopwords))
    return len(prompt_tokens & response_tokens) / len(prompt_tokens)


def _scale_without_centering(values: pd.Series) -> tuple[pd.Series, float]:
    arr = values.astype(float).to_numpy()
    sd = float(np.std(arr, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index), sd
    return pd.Series(arr / sd, index=values.index), sd


def build_helpsteer2_preference_frame(
    df: pd.DataFrame,
    *,
    keep_stopwords_for_coverage: bool = False,
) -> tuple[pd.DataFrame, dict]:
    required = {PROMPT_COL, RESPONSE_1_COL, RESPONSE_2_COL, Y_COL}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required HelpSteer2 preference columns: {missing}")

    out = df.copy()
    out = out.dropna(subset=[PROMPT_COL, RESPONSE_1_COL, RESPONSE_2_COL, Y_COL]).reset_index(drop=True)
    out[PROMPT_COL] = out[PROMPT_COL].astype(str)
    out[RESPONSE_1_COL] = out[RESPONSE_1_COL].astype(str)
    out[RESPONSE_2_COL] = out[RESPONSE_2_COL].astype(str)
    out[Y_COL] = out[Y_COL].astype(float)
    if SPLIT_COL not in out.columns:
        out[SPLIT_COL] = "all"
    out[SPLIT_COL] = out[SPLIT_COL].astype(str)

    out["sample_id"] = np.arange(len(out), dtype=int)
    out["y_preference_strength"] = out[Y_COL]

    len_1 = out[RESPONSE_1_COL].map(token_count).astype(float)
    len_2 = out[RESPONSE_2_COL].map(token_count).astype(float)
    sent_1 = out[RESPONSE_1_COL].map(sentence_count).astype(float)
    sent_2 = out[RESPONSE_2_COL].map(sentence_count).astype(float)
    cov_1 = [
        prompt_coverage(prompt, response, keep_stopwords=keep_stopwords_for_coverage)
        for prompt, response in zip(out[PROMPT_COL], out[RESPONSE_1_COL])
    ]
    cov_2 = [
        prompt_coverage(prompt, response, keep_stopwords=keep_stopwords_for_coverage)
        for prompt, response in zip(out[PROMPT_COL], out[RESPONSE_2_COL])
    ]
    fmt_1 = out[RESPONSE_1_COL].map(has_structured_format).astype(float)
    fmt_2 = out[RESPONSE_2_COL].map(has_structured_format).astype(float)

    out["response_1_word_count"] = len_1
    out["response_2_word_count"] = len_2
    out["response_1_sentence_count"] = sent_1
    out["response_2_sentence_count"] = sent_2
    out["response_1_prompt_coverage"] = cov_1
    out["response_2_prompt_coverage"] = cov_2
    out["response_1_format"] = fmt_1
    out["response_2_format"] = fmt_2

    out["delta_log_length"] = np.log1p(len_2) - np.log1p(len_1)
    out["delta_prompt_coverage"] = out["response_2_prompt_coverage"] - out["response_1_prompt_coverage"]
    out["delta_format"] = out["response_2_format"] - out["response_1_format"]
    out["delta_log_sentences"] = np.log1p(sent_2) - np.log1p(sent_1)

    scale_metadata: dict[str, float] = {}
    for column in ["delta_log_length", "delta_prompt_coverage", "delta_format", "delta_log_sentences"]:
        scaled, sd = _scale_without_centering(out[column])
        out[f"{column}_scale"] = scaled
        scale_metadata[column] = sd

    out["text_1"] = "Prompt:\n" + out[PROMPT_COL] + "\n\nResponse:\n" + out[RESPONSE_1_COL]
    out["text_2"] = "Prompt:\n" + out[PROMPT_COL] + "\n\nResponse:\n" + out[RESPONSE_2_COL]

    summary = {
        "n_rows": int(len(out)),
        "split_counts": out[SPLIT_COL].value_counts(dropna=False).sort_index().to_dict(),
        "y_min": float(out["y_preference_strength"].min()) if len(out) else math.nan,
        "y_max": float(out["y_preference_strength"].max()) if len(out) else math.nan,
        "y_mean": float(out["y_preference_strength"].mean()) if len(out) else math.nan,
        "feature_scale_without_centering_sd": scale_metadata,
        "feature_definitions": {
            "delta_log_length": "log(1 + tokens(response_2)) - log(1 + tokens(response_1))",
            "delta_prompt_coverage": "lexical prompt-token coverage(response_2) - lexical prompt-token coverage(response_1)",
            "delta_format": "1{response_2 has bullet/numbered/markdown/list structure} - 1{response_1 has structure}",
            "delta_log_sentences": "log(1 + sentence_count(response_2)) - log(1 + sentence_count(response_1))",
        },
        "sign_convention": "positive y means response_2 is preferred over response_1; all delta features are response_2 minus response_1",
    }
    return out, summary


def _read_local_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    suffixes = [item.lower() for item in path.suffixes]
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffixes[-2:] == [".jsonl", ".gz"]:
        return pd.read_json(path, lines=True, compression="gzip")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported local source format: {path}")


def load_helpsteer2_preference(*, local_source: str | Path | None = None) -> pd.DataFrame:
    if local_source is not None:
        return _read_local_table(local_source)
    try:
        from datasets import load_dataset
        dataset = load_dataset("nvidia/HelpSteer2", data_dir="preference")["train"]
        return dataset.to_pandas()
    except Exception:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(HELPSTEER2_PREFERENCE_URL, timeout=120) as response:
            payload = response.read()
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as gz:
            return pd.read_json(gz, lines=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HelpSteer2 preference pairs for coefficient inference.")
    parser.add_argument("--local-source", type=str, default=None, help="Optional local csv/json/jsonl/parquet source.")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "validation"],
        help="Optional split filter after loading the preference table.",
    )
    parser.add_argument(
        "--coverage-keep-stopwords",
        action="store_true",
        help="Keep stopwords in lexical prompt coverage. Default removes sklearn English stopwords.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_helpsteer2_preference(local_source=args.local_source)
    if args.split != "all":
        if SPLIT_COL not in raw.columns:
            raise ValueError("--split was requested but source has no split column")
        raw = raw.loc[raw[SPLIT_COL].astype(str) == args.split].copy()
    if args.max_rows is not None:
        raw = raw.head(args.max_rows).copy()

    frame, summary = build_helpsteer2_preference_frame(
        raw,
        keep_stopwords_for_coverage=args.coverage_keep_stopwords,
    )
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    summary_path = Path(args.summary_json) if args.summary_json else output.with_suffix(".summary.json")
    write_json(summary_path, summary)

    print(f"wrote {output} rows={len(frame)}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True)[:3000])


if __name__ == "__main__":
    main()
