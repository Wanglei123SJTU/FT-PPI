from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    zipf_frequency = None

BINARY_FEATURES = ["QUESTION", "NUMERIC"]
CONTINUOUS_FEATURES = ["SIMPLICITY", "COMMON", "LENGTH"]
FEATURES = [*BINARY_FEATURES, *CONTINUOUS_FEATURES]

COMMON_WORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "back",
    "be",
    "because",
    "been",
    "before",
    "being",
    "best",
    "better",
    "big",
    "but",
    "by",
    "can",
    "could",
    "day",
    "did",
    "do",
    "does",
    "down",
    "even",
    "every",
    "first",
    "for",
    "from",
    "get",
    "give",
    "go",
    "good",
    "got",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "know",
    "like",
    "look",
    "made",
    "make",
    "man",
    "many",
    "may",
    "me",
    "more",
    "most",
    "much",
    "my",
    "new",
    "no",
    "not",
    "now",
    "of",
    "off",
    "on",
    "one",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "people",
    "really",
    "right",
    "said",
    "say",
    "see",
    "she",
    "should",
    "so",
    "some",
    "take",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "thing",
    "this",
    "those",
    "time",
    "to",
    "too",
    "two",
    "up",
    "us",
    "use",
    "very",
    "want",
    "was",
    "way",
    "we",
    "well",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "woman",
    "women",
    "would",
    "you",
    "your",
}

NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
}

INTERROGATIVE_WORDS = {
    "what",
    "why",
    "who",
    "whom",
    "whose",
    "which",
    "when",
    "where",
    "how",
    "is",
    "are",
    "am",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "have",
    "has",
    "had",
}

QUESTION_START_WORDS = INTERROGATIVE_WORDS

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[,.]\d+)*(?:st|nd|rd|th)?", re.IGNORECASE)
LETTER_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")
VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(str(text))]


def _word_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if LETTER_RE.search(token)]


def count_syllables(word: str) -> int:
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 0
    groups = VOWEL_GROUP_RE.findall(cleaned)
    count = len(groups)
    if cleaned.endswith("e") and count > 1 and not cleaned.endswith(("le", "ye")):
        count -= 1
    return max(count, 1)


def sentence_count(text: str) -> int:
    parts = [part for part in re.split(r"[.!?]+", str(text)) if part.strip()]
    return max(len(parts), 1)


def _safe_share(count: float, denom: float) -> float:
    return float(count / denom) if denom else 0.0


def common_word_score(words: list[str]) -> float:
    if not words:
        return 0.0
    if zipf_frequency is not None:
        return float(np.mean([zipf_frequency(word, "en") for word in words]))
    return _safe_share(sum(1 for word in words if word in COMMON_WORDS), len(words))


def raw_headline_features(text: str) -> dict[str, float]:
    tokens = tokenize(text)
    words = _word_tokens(tokens)
    n_words = len(words)
    n_sentences = sentence_count(text)
    clean_word_lengths = [len(re.sub(r"[^a-z]", "", word)) for word in words]
    avg_word_length = _safe_share(sum(clean_word_lengths), n_words)
    syllables = sum(count_syllables(word) for word in words)
    words_per_sentence = _safe_share(n_words, n_sentences)
    syllables_per_word = _safe_share(syllables, n_words)
    fk_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59 if n_words else 0.0

    big_words = sum(1 for word in words if len(re.sub(r"[^a-z]", "", word)) >= 7)
    common_words = sum(1 for word in words if word in COMMON_WORDS)
    short_words = sum(1 for word in words if len(re.sub(r"[^a-z]", "", word)) <= 6)
    number_tokens = sum(1 for token in tokens if DIGIT_RE.search(token) or token in NUMBER_WORDS)
    interrogatives = sum(1 for word in words if word in INTERROGATIVE_WORDS)
    has_question = int(str(text).count("?") > 0 or (len(words) > 0 and words[0] in QUESTION_START_WORDS))

    return {
        "word_count": float(n_words),
        "sentence_count": float(n_sentences),
        "avg_word_length": float(avg_word_length),
        "words_per_sentence": float(words_per_sentence),
        "fk_grade": float(fk_grade),
        "big_word_share": _safe_share(big_words, n_words),
        "common_word_share": _safe_share(common_words, n_words),
        "common_word_score": common_word_score(words),
        "familiarity_proxy": 0.5 * _safe_share(common_words, n_words) + 0.5 * _safe_share(short_words, n_words),
        "number_share": _safe_share(number_tokens, n_words),
        "has_numeric": float(number_tokens > 0),
        "has_question": float(has_question),
        "question_mark_count": float(str(text).count("?")),
        "interrogative_share": _safe_share(interrogatives, n_words),
        "number_count": float(number_tokens),
        "interrogative_count": float(interrogatives),
    }


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    std = float(numeric.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(np.zeros(len(numeric)), index=series.index)
    return (numeric - float(numeric.mean())) / std


def construct_arm_features(arms: pd.DataFrame) -> pd.DataFrame:
    raw = pd.DataFrame([raw_headline_features(text) for text in arms["headline"]])
    out = pd.concat([arms[["test_id", "arm_id", "headline"]].reset_index(drop=True), raw], axis=1)

    out["rev_word_count"] = -out["word_count"]
    out["rev_avg_word_length"] = -out["avg_word_length"]
    out["rev_words_per_sentence"] = -out["words_per_sentence"]
    out["rev_fk_grade"] = -out["fk_grade"]
    out["rev_big_word_share"] = -out["big_word_share"]

    z_cols = [
        "word_count",
        "rev_word_count",
        "rev_avg_word_length",
        "rev_words_per_sentence",
        "rev_fk_grade",
        "rev_big_word_share",
        "common_word_score",
        "common_word_share",
        "familiarity_proxy",
        "number_share",
        "question_mark_count",
        "interrogative_share",
    ]
    for col in z_cols:
        out[f"{col}_z"] = zscore(out[col])

    out["QUESTION"] = out["has_question"]
    out["NUMERIC"] = out["has_numeric"]
    out["SIMPLICITY"] = out[["rev_word_count_z", "rev_avg_word_length_z", "rev_words_per_sentence_z"]].mean(axis=1)
    out["COMMON"] = out["common_word_score"]
    out["LENGTH"] = out["word_count"]
    return out


def add_pair_differences(pairs: pd.DataFrame, arm_features: pd.DataFrame) -> pd.DataFrame:
    keep = ["test_id", "arm_id", *FEATURES]
    a = arm_features[keep].rename(columns={"arm_id": "arm_id_a", **{feature: f"{feature}_a" for feature in FEATURES}})
    b = arm_features[keep].rename(columns={"arm_id": "arm_id_b", **{feature: f"{feature}_b" for feature in FEATURES}})
    merged = pairs.merge(a, on=["test_id", "arm_id_a"], how="left").merge(b, on=["test_id", "arm_id_b"], how="left")
    missing = merged[[f"{feature}_a" for feature in FEATURES] + [f"{feature}_b" for feature in FEATURES]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"Missing text features for {int(missing.sum())} pair rows")
    merged["intercept"] = 1.0
    for feature in FEATURES:
        raw_delta = merged[f"{feature}_a"] - merged[f"{feature}_b"]
        merged[f"delta_{feature}_raw"] = raw_delta
        if feature in CONTINUOUS_FEATURES:
            merged[f"delta_{feature}"] = zscore(raw_delta)
        else:
            merged[f"delta_{feature}"] = raw_delta
    return merged


def summarize_features(arm_features: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, Any]:
    return {
        "arm_rows": int(len(arm_features)),
        "pair_rows": int(len(pairs)),
        "features": FEATURES,
        "binary_features": BINARY_FEATURES,
        "continuous_pair_zscored_features": CONTINUOUS_FEATURES,
        "excluded_feature": "LIWC/Text-Analyzer-dependent constructs, including EI and the full paper versions of RE/CW/YOU.",
        "common_word_source": "wordfreq.zipf_frequency" if zipf_frequency is not None else "built-in high-frequency fallback list",
        "construct_notes": {
            "QUESTION": "Binary question-headline cue: question mark or interrogative/auxiliary first word.",
            "NUMERIC": "Binary numeric cue: any digit token or written number/ordinal token.",
            "SIMPLICITY": "Transparent reading-ease proxy: arm-level average of reversed standardized word count, average word length, and words per sentence; the pair difference is z-scored for regression.",
            "COMMON": "Common-word usage: arm-level mean Zipf frequency from wordfreq when available, otherwise share of a built-in frequent-word list; the pair difference is z-scored for regression.",
            "LENGTH": "Headline length: arm-level word count; the pair difference is z-scored for regression.",
        },
        "delta_summary": {
            feature: {
                "mean": float(pairs[f"delta_{feature}"].mean()),
                "std": float(pairs[f"delta_{feature}"].std(ddof=0)),
                "nonzero_share": float((pairs[f"delta_{feature}"] != 0).mean()),
            }
            for feature in FEATURES
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct Upworthy headline features for pair-level M-estimation.")
    parser.add_argument("--arms", type=Path, default=Path("artifacts/upworthy_m_estimation/simple_lola_like/ctr_arms_lola_like.csv"))
    parser.add_argument("--pairs", type=Path, default=Path("artifacts/upworthy_m_estimation/simple_lola_like/pairs_one_per_test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/upworthy_m_estimation/simple_lola_like"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arms = pd.read_csv(args.arms)
    pairs = pd.read_csv(args.pairs)
    arm_features = construct_arm_features(arms)
    pair_features = add_pair_differences(pairs, arm_features)
    arm_features.to_csv(args.output_dir / "headline_arm_text_features.csv", index=False)
    pair_features.to_csv(args.output_dir / "pairs_with_text_features.csv", index=False)
    summary = summarize_features(arm_features, pair_features)
    (args.output_dir / "text_feature_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
