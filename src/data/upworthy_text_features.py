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

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - optional dependency fallback
    SentimentIntensityAnalyzer = None

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except ImportError:  # pragma: no cover - sklearn is a project dependency
    ENGLISH_STOP_WORDS = frozenset()

BINARY_FEATURES = [
    "QUESTION",
    "NUMERIC",
    "NEGATION",
    "SECOND_PERSON",
    "FIRST_PERSON",
    "EXCLAMATION",
    "HAS_QUOTES",
]
CONTINUOUS_FEATURES = [
    "SIMPLICITY",
    "COMMON",
    "LENGTH",
    "READING_EASE",
    "FK_GRADE",
    "AVG_WORD_LENGTH",
    "CHAR_LENGTH",
    "LONG_WORD_SHARE",
    "COMMON_CONTENT",
    "STOPWORD_SHARE",
    "UNIQUE_WORD_SHARE",
    "VADER_POS",
    "VADER_NEG",
    "VADER_COMPOUND",
    "SENTIMENT_EXTREMITY",
    "SENTIMENT_INTENSITY",
    "QUESTION_MARKS",
    "INTERROGATIVE",
    "NUMBER_SHARE",
    "NEGATION_SHARE",
    "SECOND_PERSON_SHARE",
    "FIRST_PERSON_SHARE",
    "WE_PRONOUN_SHARE",
    "THEY_PRONOUN_SHARE",
    "EXCLAMATION_MARKS",
    "CAPS_SHARE",
]
FEATURES = [*BINARY_FEATURES, *CONTINUOUS_FEATURES]

FEATURE_METADATA: dict[str, dict[str, str]] = {
    "QUESTION": {
        "source": "transparent rule",
        "note": "Question mark or interrogative/auxiliary first word.",
    },
    "NUMERIC": {
        "source": "transparent rule",
        "note": "Any digit token or written number/ordinal token.",
    },
    "NEGATION": {
        "source": "transparent rule",
        "note": "Any standard English negation token.",
    },
    "SECOND_PERSON": {
        "source": "transparent rule",
        "note": "Any second-person pronoun token.",
    },
    "FIRST_PERSON": {
        "source": "transparent rule",
        "note": "Any first-person singular/plural pronoun token.",
    },
    "EXCLAMATION": {
        "source": "transparent rule",
        "note": "Headline contains an exclamation mark.",
    },
    "HAS_QUOTES": {
        "source": "transparent rule",
        "note": "Headline contains single or double quotes.",
    },
    "SIMPLICITY": {
        "source": "transparent composite",
        "note": "Average of reversed standardized word count, average word length, and words per sentence.",
    },
    "COMMON": {
        "source": "wordfreq.zipf_frequency",
        "note": "Mean Zipf frequency over all word tokens when wordfreq is available.",
    },
    "LENGTH": {
        "source": "transparent rule",
        "note": "Word count.",
    },
    "READING_EASE": {
        "source": "standard Flesch Reading Ease formula",
        "note": "Offline implementation using token, sentence, and syllable counts.",
    },
    "FK_GRADE": {
        "source": "standard Flesch-Kincaid grade formula",
        "note": "Offline implementation using token, sentence, and syllable counts.",
    },
    "AVG_WORD_LENGTH": {
        "source": "transparent rule",
        "note": "Average alphabetic word length.",
    },
    "CHAR_LENGTH": {
        "source": "transparent rule",
        "note": "Number of characters in the headline string.",
    },
    "LONG_WORD_SHARE": {
        "source": "transparent rule",
        "note": "Share of word tokens with at least 7 letters.",
    },
    "COMMON_CONTENT": {
        "source": "wordfreq.zipf_frequency + sklearn ENGLISH_STOP_WORDS",
        "note": "Mean Zipf frequency over non-stopword tokens.",
    },
    "STOPWORD_SHARE": {
        "source": "sklearn ENGLISH_STOP_WORDS",
        "note": "Share of word tokens in the sklearn English stopword list.",
    },
    "UNIQUE_WORD_SHARE": {
        "source": "transparent rule",
        "note": "Unique word-token share.",
    },
    "VADER_POS": {
        "source": "vaderSentiment",
        "note": "VADER positive sentiment score.",
    },
    "VADER_NEG": {
        "source": "vaderSentiment",
        "note": "VADER negative sentiment score.",
    },
    "VADER_COMPOUND": {
        "source": "vaderSentiment",
        "note": "VADER compound sentiment score.",
    },
    "SENTIMENT_EXTREMITY": {
        "source": "vaderSentiment",
        "note": "Absolute VADER compound score.",
    },
    "SENTIMENT_INTENSITY": {
        "source": "vaderSentiment",
        "note": "VADER positive plus negative score.",
    },
    "QUESTION_MARKS": {
        "source": "transparent rule",
        "note": "Question mark count.",
    },
    "INTERROGATIVE": {
        "source": "transparent rule",
        "note": "Share of word tokens in a standard interrogative/auxiliary list.",
    },
    "NUMBER_SHARE": {
        "source": "transparent rule",
        "note": "Number token share.",
    },
    "NEGATION_SHARE": {
        "source": "transparent rule",
        "note": "Negation token share.",
    },
    "SECOND_PERSON_SHARE": {
        "source": "transparent rule",
        "note": "Second-person pronoun token share.",
    },
    "FIRST_PERSON_SHARE": {
        "source": "transparent rule",
        "note": "First-person pronoun token share.",
    },
    "WE_PRONOUN_SHARE": {
        "source": "transparent rule",
        "note": "First-person plural pronoun token share.",
    },
    "THEY_PRONOUN_SHARE": {
        "source": "transparent rule",
        "note": "Third-person plural pronoun token share.",
    },
    "EXCLAMATION_MARKS": {
        "source": "transparent rule",
        "note": "Exclamation mark count.",
    },
    "CAPS_SHARE": {
        "source": "transparent rule",
        "note": "Uppercase share among alphabetic characters.",
    },
}

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

NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "none",
    "nobody",
    "nothing",
    "neither",
    "nor",
    "cannot",
    "can't",
    "won't",
    "dont",
    "don't",
    "doesnt",
    "doesn't",
    "didnt",
    "didn't",
    "isnt",
    "isn't",
    "arent",
    "aren't",
    "wasnt",
    "wasn't",
    "werent",
    "weren't",
    "hasnt",
    "hasn't",
    "havent",
    "haven't",
    "hadnt",
    "hadn't",
    "without",
}

SECOND_PERSON_WORDS = {"you", "your", "yours", "yourself", "yourselves", "u"}
FIRST_PERSON_WORDS = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
}
WE_PRONOUN_WORDS = {"we", "us", "our", "ours", "ourselves"}
THEY_PRONOUN_WORDS = {"they", "them", "their", "theirs", "themselves"}

_VADER_ANALYZER = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer is not None else None

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


def content_words(words: list[str]) -> list[str]:
    if not ENGLISH_STOP_WORDS:
        return list(words)
    return [word for word in words if word not in ENGLISH_STOP_WORDS]


def vader_scores(text: str) -> dict[str, float]:
    if _VADER_ANALYZER is None:
        return {"pos": 0.0, "neg": 0.0, "neu": 0.0, "compound": 0.0}
    scores = _VADER_ANALYZER.polarity_scores(str(text))
    return {key: float(scores.get(key, 0.0)) for key in ["pos", "neg", "neu", "compound"]}


def raw_headline_features(text: str) -> dict[str, float]:
    text_value = str(text)
    tokens = tokenize(text)
    words = _word_tokens(tokens)
    n_words = len(words)
    n_sentences = sentence_count(text)
    clean_word_lengths = [len(re.sub(r"[^a-z]", "", word)) for word in words]
    avg_word_length = _safe_share(sum(clean_word_lengths), n_words)
    syllables = sum(count_syllables(word) for word in words)
    words_per_sentence = _safe_share(n_words, n_sentences)
    syllables_per_word = _safe_share(syllables, n_words)
    fk_grade_fallback = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59 if n_words else 0.0
    reading_ease_fallback = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word if n_words else 0.0
    fk_grade = float(fk_grade_fallback)
    reading_ease = float(reading_ease_fallback)

    big_words = sum(1 for word in words if len(re.sub(r"[^a-z]", "", word)) >= 7)
    common_words = sum(1 for word in words if word in COMMON_WORDS)
    short_words = sum(1 for word in words if len(re.sub(r"[^a-z]", "", word)) <= 6)
    number_tokens = sum(1 for token in tokens if DIGIT_RE.search(token) or token in NUMBER_WORDS)
    interrogatives = sum(1 for word in words if word in INTERROGATIVE_WORDS)
    negations = sum(1 for word in words if word in NEGATION_WORDS)
    second_person = sum(1 for word in words if word in SECOND_PERSON_WORDS)
    first_person = sum(1 for word in words if word in FIRST_PERSON_WORDS)
    we_pronouns = sum(1 for word in words if word in WE_PRONOUN_WORDS)
    they_pronouns = sum(1 for word in words if word in THEY_PRONOUN_WORDS)
    stopwords = sum(1 for word in words if word in ENGLISH_STOP_WORDS)
    content = content_words(words)
    has_question = int(text_value.count("?") > 0 or (len(words) > 0 and words[0] in QUESTION_START_WORDS))
    quote_count = text_value.count('"') + text_value.count("'")
    exclamation_count = text_value.count("!")
    letters = LETTER_RE.findall(text_value)
    caps_count = sum(1 for char in letters if char.isupper())
    sentiment = vader_scores(text_value)

    return {
        "word_count": float(n_words),
        "char_count": float(len(text_value)),
        "sentence_count": float(n_sentences),
        "avg_word_length": float(avg_word_length),
        "words_per_sentence": float(words_per_sentence),
        "fk_grade": float(fk_grade),
        "reading_ease": float(reading_ease),
        "big_word_share": _safe_share(big_words, n_words),
        "common_word_share": _safe_share(common_words, n_words),
        "common_word_score": common_word_score(words),
        "common_content_word_score": common_word_score(content),
        "familiarity_proxy": 0.5 * _safe_share(common_words, n_words) + 0.5 * _safe_share(short_words, n_words),
        "stopword_share": _safe_share(stopwords, n_words),
        "unique_word_share": _safe_share(len(set(words)), n_words),
        "number_share": _safe_share(number_tokens, n_words),
        "has_numeric": float(number_tokens > 0),
        "has_question": float(has_question),
        "has_negation": float(negations > 0),
        "has_second_person": float(second_person > 0),
        "has_first_person": float(first_person > 0),
        "has_exclamation": float(exclamation_count > 0),
        "has_quotes": float(quote_count > 0),
        "question_mark_count": float(text_value.count("?")),
        "interrogative_share": _safe_share(interrogatives, n_words),
        "negation_share": _safe_share(negations, n_words),
        "second_person_share": _safe_share(second_person, n_words),
        "first_person_share": _safe_share(first_person, n_words),
        "we_pronoun_share": _safe_share(we_pronouns, n_words),
        "they_pronoun_share": _safe_share(they_pronouns, n_words),
        "exclamation_count": float(exclamation_count),
        "quote_count": float(quote_count),
        "caps_share": _safe_share(caps_count, len(letters)),
        "vader_pos": sentiment["pos"],
        "vader_neg": sentiment["neg"],
        "vader_neu": sentiment["neu"],
        "vader_compound": sentiment["compound"],
        "sentiment_extremity": abs(sentiment["compound"]),
        "sentiment_intensity": sentiment["pos"] + sentiment["neg"],
        "number_count": float(number_tokens),
        "interrogative_count": float(interrogatives),
        "negation_count": float(negations),
        "second_person_count": float(second_person),
        "first_person_count": float(first_person),
    }


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    std = float(numeric.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(np.zeros(len(numeric)), index=series.index)
    return (numeric - float(numeric.mean())) / std


def antisymmetric_scale(series: pd.Series) -> tuple[pd.Series, float]:
    """Scale pair differences without centering, preserving A/B sign symmetry."""
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    scale = float(np.sqrt(np.mean(np.square(numeric))))
    if not np.isfinite(scale) or scale == 0.0:
        return pd.Series(np.zeros(len(numeric)), index=series.index), 1.0
    return numeric / scale, scale


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
        "char_count",
        "rev_word_count",
        "rev_avg_word_length",
        "rev_words_per_sentence",
        "rev_fk_grade",
        "rev_big_word_share",
        "reading_ease",
        "common_word_score",
        "common_content_word_score",
        "common_word_share",
        "familiarity_proxy",
        "stopword_share",
        "unique_word_share",
        "number_share",
        "question_mark_count",
        "interrogative_share",
        "negation_share",
        "second_person_share",
        "first_person_share",
        "we_pronoun_share",
        "they_pronoun_share",
        "exclamation_count",
        "caps_share",
        "vader_pos",
        "vader_neg",
        "vader_compound",
        "sentiment_extremity",
        "sentiment_intensity",
    ]
    for col in z_cols:
        out[f"{col}_z"] = zscore(out[col])

    out["QUESTION"] = out["has_question"]
    out["NUMERIC"] = out["has_numeric"]
    out["NEGATION"] = out["has_negation"]
    out["SECOND_PERSON"] = out["has_second_person"]
    out["FIRST_PERSON"] = out["has_first_person"]
    out["EXCLAMATION"] = out["has_exclamation"]
    out["HAS_QUOTES"] = out["has_quotes"]
    out["SIMPLICITY"] = out[["rev_word_count_z", "rev_avg_word_length_z", "rev_words_per_sentence_z"]].mean(axis=1)
    out["COMMON"] = out["common_word_score"]
    out["LENGTH"] = out["word_count"]
    out["READING_EASE"] = out["reading_ease"]
    out["FK_GRADE"] = out["fk_grade"]
    out["AVG_WORD_LENGTH"] = out["avg_word_length"]
    out["CHAR_LENGTH"] = out["char_count"]
    out["LONG_WORD_SHARE"] = out["big_word_share"]
    out["COMMON_CONTENT"] = out["common_content_word_score"]
    out["STOPWORD_SHARE"] = out["stopword_share"]
    out["UNIQUE_WORD_SHARE"] = out["unique_word_share"]
    out["VADER_POS"] = out["vader_pos"]
    out["VADER_NEG"] = out["vader_neg"]
    out["VADER_COMPOUND"] = out["vader_compound"]
    out["SENTIMENT_EXTREMITY"] = out["sentiment_extremity"]
    out["SENTIMENT_INTENSITY"] = out["sentiment_intensity"]
    out["QUESTION_MARKS"] = out["question_mark_count"]
    out["INTERROGATIVE"] = out["interrogative_share"]
    out["NUMBER_SHARE"] = out["number_share"]
    out["NEGATION_SHARE"] = out["negation_share"]
    out["SECOND_PERSON_SHARE"] = out["second_person_share"]
    out["FIRST_PERSON_SHARE"] = out["first_person_share"]
    out["WE_PRONOUN_SHARE"] = out["we_pronoun_share"]
    out["THEY_PRONOUN_SHARE"] = out["they_pronoun_share"]
    out["EXCLAMATION_MARKS"] = out["exclamation_count"]
    out["CAPS_SHARE"] = out["caps_share"]
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
            scaled_delta, scale = antisymmetric_scale(raw_delta)
            merged[f"delta_{feature}"] = scaled_delta
            merged[f"delta_{feature}_scale"] = scale
        else:
            merged[f"delta_{feature}"] = raw_delta
    return merged


def drop_existing_text_feature_columns(pairs: pd.DataFrame) -> pd.DataFrame:
    """Drop stale feature columns before rebuilding them from pair headlines."""
    drop_cols = ["intercept"]
    for feature in FEATURES:
        drop_cols.extend(
            [
                f"{feature}_a",
                f"{feature}_b",
                f"delta_{feature}",
                f"delta_{feature}_raw",
                f"delta_{feature}_scale",
            ]
        )
    return pairs.drop(columns=[col for col in drop_cols if col in pairs.columns])


def construct_arm_features_from_pair_headlines(pairs: pd.DataFrame) -> pd.DataFrame:
    required = {"test_id", "arm_id_a", "arm_id_b", "headline_a", "headline_b"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError("pair-level input is missing required columns: " + ", ".join(missing))
    arms_a = pairs[["test_id", "arm_id_a", "headline_a"]].rename(
        columns={"arm_id_a": "arm_id", "headline_a": "headline"}
    )
    arms_b = pairs[["test_id", "arm_id_b", "headline_b"]].rename(
        columns={"arm_id_b": "arm_id", "headline_b": "headline"}
    )
    arms = (
        pd.concat([arms_a, arms_b], ignore_index=True)
        .drop_duplicates(["test_id", "arm_id"], keep="first")
        .reset_index(drop=True)
    )
    return construct_arm_features(arms)


def add_pair_features_from_headlines(pairs: pd.DataFrame) -> pd.DataFrame:
    arm_features = construct_arm_features_from_pair_headlines(pairs)
    clean_pairs = drop_existing_text_feature_columns(pairs)
    return add_pair_differences(clean_pairs, arm_features)


def summarize_features(arm_features: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, Any]:
    return {
        "arm_rows": int(len(arm_features)),
        "pair_rows": int(len(pairs)),
        "features": FEATURES,
        "binary_features": BINARY_FEATURES,
        "continuous_pair_scaled_features": CONTINUOUS_FEATURES,
        "continuous_pair_scaling_rule": "delta_feature = raw_delta / sqrt(mean(raw_delta^2)); no empirical centering, so swapping A/B maps delta to -delta.",
        "excluded_feature": "LIWC/Text-Analyzer-dependent constructs and paper dictionaries not available as machine-readable word-list files in the local OSF download.",
        "common_word_source": "wordfreq.zipf_frequency" if zipf_frequency is not None else "built-in high-frequency fallback list",
        "feature_sources": {
            "wordfreq": zipf_frequency is not None,
            "offline_flesch_formula": True,
            "vaderSentiment": _VADER_ANALYZER is not None,
            "sklearn_stopwords": bool(ENGLISH_STOP_WORDS),
        },
        "construct_notes": FEATURE_METADATA,
        "delta_summary": {
            feature: {
                "mean": float(pairs[f"delta_{feature}"].mean()),
                "std": float(pairs[f"delta_{feature}"].std(ddof=0)),
                "nonzero_share": float((pairs[f"delta_{feature}"] != 0).mean()),
                "raw_scale": (
                    float(pairs[f"delta_{feature}_scale"].iloc[0])
                    if f"delta_{feature}_scale" in pairs and len(pairs) > 0
                    else None
                ),
            }
            for feature in FEATURES
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct Upworthy headline features for pair-level M-estimation.")
    parser.add_argument("--arms", type=Path, default=Path("artifacts/upworthy_m_estimation/simple_lola_like/ctr_arms_lola_like.csv"))
    parser.add_argument("--pairs", type=Path, default=Path("artifacts/upworthy_m_estimation/simple_lola_like/pairs_one_per_test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/upworthy_m_estimation/simple_lola_like"))
    parser.add_argument("--from-pair-headlines", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs)
    if args.from_pair_headlines:
        arm_features = construct_arm_features_from_pair_headlines(pairs)
        pair_features = add_pair_features_from_headlines(pairs)
    else:
        arms = pd.read_csv(args.arms)
        arm_features = construct_arm_features(arms)
        pair_features = add_pair_differences(pairs, arm_features)
    output_csv = args.output_csv or (args.output_dir / "pairs_with_text_features.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    arm_features.to_csv(args.output_dir / "headline_arm_text_features.csv", index=False)
    pair_features.to_csv(output_csv, index=False)
    summary = summarize_features(arm_features, pair_features)
    summary["output_csv"] = str(output_csv)
    (args.output_dir / "text_feature_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
