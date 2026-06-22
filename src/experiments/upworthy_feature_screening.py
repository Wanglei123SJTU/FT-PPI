from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from src.data.upworthy_text_features import (
    COMMON_WORDS,
    DIGIT_RE,
    INTERROGATIVE_WORDS,
    NUMBER_WORDS,
    QUESTION_START_WORDS,
    count_syllables,
    tokenize,
)
from src.formatting import dataframe_to_markdown

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - dependency is in requirements, fallback keeps diagnostics usable.
    SentimentIntensityAnalyzer = None

try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover
    zipf_frequency = None


Y_COL = "y_logit_ctr_diff"
TEXT_A_COL = "headline_a"
TEXT_B_COL = "headline_b"
DEFAULT_OUTPUT_DIR = Path("artifacts/upworthy_m_estimation/feature_screening_v2")
DEFAULT_CI_BUDGETS = [500, 1000, 1500, 3000]

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[,.]\d+)*(?:st|nd|rd|th)?", re.IGNORECASE)
LETTER_RE = re.compile(r"[A-Za-z]")
SENTENCE_RE = re.compile(r"[.!?]+")
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves", "u"}
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
NEGATIONS = {"no", "not", "never", "none", "nothing", "nobody", "nowhere", "neither", "nor", "without", "can't", "won't", "don't", "doesn't", "isn't", "aren't"}
DEMONSTRATIVES = {"this", "that", "these", "those"}
SUPERLATIVES = {"best", "worst", "most", "least", "biggest", "smallest", "greatest", "largest", "easiest", "hardest"}
URGENCY = {"now", "today", "before", "after", "soon", "urgent", "immediately", "finally"}
SURPRISE = {"surprising", "shocking", "unbelievable", "amazing", "weird", "secret", "truth", "actually", "really"}
CURIOSITY = {
    "why",
    "how",
    "what",
    "this",
    "these",
    "that",
    "reason",
    "reasons",
    "secret",
    "secrets",
    "happens",
    "happened",
    "happen",
    "thing",
    "things",
}


@dataclass(frozen=True)
class FeatureInfo:
    feature: str
    raw_column: str
    family: str
    kind: str
    note: str
    priority: int = 2


def _word_tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(str(text)) if LETTER_RE.search(token)]


def _content_tokens(text: str) -> list[str]:
    return [token for token in _word_tokens(text) if token not in ENGLISH_STOP_WORDS]


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _sentence_count(text: str) -> int:
    parts = [part for part in SENTENCE_RE.split(str(text)) if part.strip()]
    return max(len(parts), 1)


def _zipf(word: str) -> float:
    if zipf_frequency is None:
        return 4.5 if word in COMMON_WORDS else 3.0
    return float(zipf_frequency(word, "en"))


def _share(tokens: list[str], vocabulary: set[str]) -> float:
    return _safe_div(sum(1 for token in tokens if token in vocabulary), len(tokens))


def _question_cue(text: str, words: list[str]) -> float:
    if "?" in str(text):
        return 1.0
    return float(bool(words) and words[0] in QUESTION_START_WORDS)


def _numeric_cue(tokens: list[str]) -> float:
    return float(any(DIGIT_RE.search(token) or token in NUMBER_WORDS for token in tokens))


def _readability(words: list[str], sentence_count: int) -> tuple[float, float]:
    if not words:
        return 0.0, 0.0
    syllables = sum(count_syllables(word) for word in words)
    words_per_sentence = len(words) / max(sentence_count, 1)
    syllables_per_word = syllables / len(words)
    reading_ease = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    fk_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    return float(reading_ease), float(fk_grade)


def _coverage(headline_tokens: list[str], context_tokens: list[str]) -> float:
    headline_set = set(headline_tokens)
    if not headline_set:
        return 0.0
    context_set = set(context_tokens)
    return float(len(headline_set & context_set) / len(headline_set))


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    denom = len(left_set | right_set)
    if denom == 0:
        return 0.0
    return float(len(left_set & right_set) / denom)


def _vader_scores(texts: pd.Series) -> pd.DataFrame:
    if SentimentIntensityAnalyzer is None:
        zeros = np.zeros(len(texts), dtype=float)
        return pd.DataFrame({"vader_compound": zeros, "vader_pos": zeros, "vader_neg": zeros, "vader_neu": zeros})
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(str(text)) for text in texts]
    return pd.DataFrame(
        {
            "vader_compound": [score["compound"] for score in scores],
            "vader_pos": [score["pos"] for score in scores],
            "vader_neg": [score["neg"] for score in scores],
            "vader_neu": [score["neu"] for score in scores],
        }
    )


def _make_context(frame: pd.DataFrame, suffix: str) -> pd.Series:
    pieces = []
    for stem in ["excerpt", "lede", "share_text"]:
        col = f"{stem}_{suffix}"
        if col in frame.columns:
            pieces.append(frame[col].fillna("").astype(str))
    if not pieces:
        return pd.Series([""] * len(frame), index=frame.index)
    out = pieces[0]
    for item in pieces[1:]:
        out = out + " " + item
    return out


def _arm_feature_frame(text: pd.Series, context: pd.Series, prefix: str) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for headline, ctx in zip(text.fillna("").astype(str), context.fillna("").astype(str)):
        tokens = tokenize(headline)
        words = _word_tokens(headline)
        content = [word for word in words if word not in ENGLISH_STOP_WORDS]
        ctx_content = _content_tokens(ctx)
        n_words = len(words)
        n_content = len(content)
        n_chars = len(str(headline))
        n_sent = _sentence_count(headline)
        word_lengths = [len(re.sub(r"[^a-z]", "", word)) for word in words]
        avg_word_length = _safe_div(sum(word_lengths), n_words)
        reading_ease, fk_grade = _readability(words, n_sent)
        zipfs = [_zipf(word) for word in words]
        content_zipfs = [_zipf(word) for word in content]
        rare_words = sum(1 for value in content_zipfs if value < 3.5)
        common_words = sum(1 for value in zipfs if value >= 4.5)
        uppercase_words = sum(1 for raw in str(headline).split() if len(raw) > 1 and raw.isupper())
        punctuation = sum(1 for ch in str(headline) if ch in "!?;:-")

        rows.append(
            {
                "question": _question_cue(headline, words),
                "numeric": _numeric_cue(tokens),
                "word_count": float(n_words),
                "log_word_count": math.log1p(n_words),
                "char_count": float(n_chars),
                "log_char_count": math.log1p(n_chars),
                "avg_word_length": float(avg_word_length),
                "sentence_count": float(n_sent),
                "reading_ease": reading_ease,
                "fk_grade": fk_grade,
                "simplicity": float(-math.log1p(n_words) - avg_word_length),
                "common_zipf": float(np.mean(zipfs)) if zipfs else 0.0,
                "content_common_zipf": float(np.mean(content_zipfs)) if content_zipfs else 0.0,
                "rare_word_share": _safe_div(rare_words, n_content),
                "common_word_share": _safe_div(common_words, n_words),
                "content_word_share": _safe_div(n_content, n_words),
                "context_coverage": _coverage(content, ctx_content),
                "context_jaccard": _jaccard(content, ctx_content),
                "second_person_share": _share(words, SECOND_PERSON),
                "has_second_person": float(any(word in SECOND_PERSON for word in words)),
                "first_person_share": _share(words, FIRST_PERSON),
                "negation_share": _share(words, NEGATIONS),
                "has_negation": float(any(word in NEGATIONS for word in words)),
                "demonstrative_share": _share(words, DEMONSTRATIVES),
                "has_demonstrative": float(any(word in DEMONSTRATIVES for word in words)),
                "superlative_share": _share(words, SUPERLATIVES),
                "curiosity_share": _share(words, CURIOSITY),
                "urgency_share": _share(words, URGENCY),
                "surprise_share": _share(words, SURPRISE),
                "question_mark_count": float(str(headline).count("?")),
                "exclamation_count": float(str(headline).count("!")),
                "colon_dash_count": float(str(headline).count(":") + str(headline).count("-")),
                "quote_count": float(str(headline).count('"') + str(headline).count("'")),
                "uppercase_word_share": _safe_div(uppercase_words, n_words),
                "punctuation_intensity": _safe_div(punctuation, max(n_chars, 1)),
            }
        )
    out = pd.DataFrame(rows, index=text.index)
    vader = _vader_scores(text)
    vader.index = out.index
    out = pd.concat([out, vader], axis=1)
    out["vader_intensity"] = out["vader_compound"].abs()
    out["vader_emotion"] = out["vader_pos"] + out["vader_neg"]
    out["curiosity_style"] = (
        out["question"]
        + out["has_second_person"]
        + out["has_demonstrative"]
        + (out["curiosity_share"] > 0).astype(float)
        + (out["surprise_share"] > 0).astype(float)
    )
    return out.add_prefix(f"{prefix}_")


def _scale_no_center(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float).fillna(0.0)
    sd = float(numeric.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.zeros(len(numeric)), index=values.index)
    return numeric / sd


def _kind(raw: pd.Series) -> str:
    unique = raw.dropna().unique()
    if len(unique) <= 5 and set(np.round(unique, 8)).issubset({-1.0, 0.0, 1.0}):
        return "ternary"
    if len(unique) <= 2:
        return "binary"
    return "continuous"


def _family(name: str) -> str:
    for key in ["length", "word_count", "char_count", "sentence", "read", "fk", "simplicity"]:
        if key in name:
            return "readability_length"
    for key in ["vader", "emotion", "negation", "surprise"]:
        if key in name:
            return "sentiment"
    for key in ["coverage", "jaccard", "context"]:
        if key in name:
            return "context_alignment"
    for key in ["common", "rare", "zipf", "content_word"]:
        if key in name:
            return "specificity_commonness"
    for key in ["question", "numeric", "second_person", "demonstrative", "curiosity", "superlative", "urgency"]:
        if key in name:
            return "curiosity_style"
    for key in ["punctuation", "colon", "quote", "uppercase", "exclamation"]:
        if key in name:
            return "format_emphasis"
    return "other"


def _note(name: str, family: str) -> str:
    if name == "context_coverage":
        return "Share of headline content words that appear in the article context; headline-message alignment."
    if name == "context_jaccard":
        return "Jaccard overlap between headline content words and article context words."
    if name.startswith("vader_"):
        return "Open-source VADER headline sentiment cue."
    if name in {"common_zipf", "content_common_zipf"}:
        return "Mean word frequency from wordfreq; higher means more common/simple language."
    if name == "rare_word_share":
        return "Share of uncommon content words; proxy for specificity or complexity."
    if name == "curiosity_style":
        return "Aggregated curiosity/clickbait-style cue from question, second person, demonstratives, curiosity, and surprise words."
    if family == "readability_length":
        return "Transparent headline length/readability cue."
    if family == "format_emphasis":
        return "Headline punctuation or emphasis cue."
    return "Deterministic headline text feature."


def _priority(name: str, family: str) -> int:
    if name in {"context_coverage", "context_jaccard", "vader_intensity", "vader_neg", "rare_word_share", "curiosity_style"}:
        return 0
    if family in {"sentiment", "context_alignment", "specificity_commonness", "curiosity_style"}:
        return 1
    if family == "readability_length":
        return 2
    return 3


def build_candidate_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_a = _make_context(frame, "a")
    context_b = _make_context(frame, "b")
    arm_a = _arm_feature_frame(frame[TEXT_A_COL], context_a, "a")
    arm_b = _arm_feature_frame(frame[TEXT_B_COL], context_b, "b")
    out = pd.concat([frame.copy(), arm_a, arm_b], axis=1)

    feature_names = [column.removeprefix("a_") for column in arm_a.columns]
    info_rows: list[FeatureInfo] = []
    for name in feature_names:
        raw_col = f"delta_{name}_raw"
        feature_col = f"delta_{name}"
        raw = out[f"a_{name}"].astype(float) - out[f"b_{name}"].astype(float)
        out[raw_col] = raw
        kind = _kind(raw)
        if kind == "continuous":
            out[feature_col] = _scale_no_center(raw)
        else:
            out[feature_col] = raw.fillna(0.0).astype(float)
        family = _family(name)
        info_rows.append(
            FeatureInfo(
                feature=feature_col,
                raw_column=raw_col,
                family=family,
                kind=kind,
                note=_note(name, family),
                priority=_priority(name, family),
            )
        )

    metadata = pd.DataFrame([info.__dict__ for info in info_rows])
    return out, metadata


def design_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(len(frame)), frame[columns].astype(float).to_numpy()])


def fit_ols(frame: pd.DataFrame, y_col: str, columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = frame[y_col].astype(float).to_numpy()
    x = design_matrix(frame, columns)
    xtx = x.T @ x
    beta = np.linalg.pinv(xtx) @ x.T @ y
    hessian = xtx / len(frame)
    residual = y - x @ beta
    return beta, hessian, residual


def if_weight(x: np.ndarray, hessian: np.ndarray, target_position: int) -> np.ndarray:
    h_inv = np.linalg.pinv(hessian)
    return x @ h_inv[target_position, :]


def ess_share(weights: np.ndarray) -> float:
    denom = float(np.sum(weights**4))
    if denom <= 0:
        return 0.0
    ess = float(np.sum(weights**2) ** 2 / denom)
    return ess / len(weights)


def vif_for_target(frame: pd.DataFrame, target: str, controls: list[str]) -> float:
    if not controls:
        return 1.0
    y = frame[target].astype(float).to_numpy()
    x = design_matrix(frame, controls)
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    pred = x @ beta
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 0:
        return float("inf")
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / sst
    if r2 >= 1:
        return float("inf")
    return float(1.0 / max(1.0 - r2, 1e-12))


def _eligible_controls(target: str, all_controls: list[str], metadata: pd.DataFrame, frame: pd.DataFrame) -> list[str]:
    family = metadata.set_index("feature").loc[target, "family"]
    meta = metadata.set_index("feature")
    controls: list[str] = []
    for control in all_controls:
        if control == target or control not in frame.columns:
            continue
        if control in meta.index and meta.loc[control, "family"] == family:
            continue
        corr = frame[[target, control]].corr().iloc[0, 1]
        if np.isfinite(corr) and abs(float(corr)) >= 0.85:
            continue
        controls.append(control)
    return controls


def _tfidf_text(frame: pd.DataFrame) -> pd.Series:
    return (
        "Headline A: "
        + frame[TEXT_A_COL].fillna("").astype(str)
        + "\nHeadline B: "
        + frame[TEXT_B_COL].fillna("").astype(str)
        + "\nContext: "
        + _make_context(frame, "a")
    )


def fit_tfidf_surrogate(
    frame: pd.DataFrame,
    *,
    train_mask: pd.Series,
    eval_mask: pd.Series,
    max_features: int,
) -> tuple[np.ndarray, dict[str, float]]:
    texts = _tfidf_text(frame)
    train_texts = texts.loc[train_mask].tolist()
    eval_texts = texts.loc[eval_mask].tolist()
    y_train = frame.loc[train_mask, Y_COL].astype(float).to_numpy()
    y_eval = frame.loc[eval_mask, Y_COL].astype(float).to_numpy()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=max_features,
        strip_accents="unicode",
        lowercase=True,
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_eval = vectorizer.transform(eval_texts)
    best: tuple[float, float, Ridge] | None = None
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha, random_state=0)
        model.fit(x_train, y_train)
        pred = model.predict(x_eval)
        mse = mean_squared_error(y_eval, pred)
        if best is None or mse < best[1]:
            best = (alpha, float(mse), model)
    assert best is not None
    alpha, mse, model = best
    pred_all = np.zeros(len(frame), dtype=float)
    matrix_all = vectorizer.transform(texts.tolist())
    pred_all[:] = model.predict(matrix_all)
    return pred_all, {"tfidf_alpha": float(alpha), "tfidf_eval_mse": float(mse), "tfidf_n_features": float(len(vectorizer.vocabulary_))}


def _safe_abs_corr(frame: pd.DataFrame, target: str, controls: list[str]) -> float:
    if not controls:
        return 0.0
    corr = frame[[target, *controls]].corr()[target].drop(target).abs()
    corr = corr[np.isfinite(corr)]
    return float(corr.max()) if not corr.empty else 0.0


def summarize_candidates(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    budgets: list[int],
    max_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    target = frame.loc[frame["split"] == "target"].copy()
    h_scale_mask = frame["split"] == "h_scale"
    target_mask = frame["split"] == "target"

    pred_tfidf, tfidf_meta = fit_tfidf_surrogate(
        frame,
        train_mask=h_scale_mask,
        eval_mask=target_mask,
        max_features=max_features,
    )
    target_pred = pred_tfidf[target_mask.to_numpy()]
    y = target[Y_COL].astype(float).to_numpy()

    core_controls = [
        "delta_question",
        "delta_numeric",
        "delta_log_word_count",
        "delta_context_coverage",
        "delta_vader_compound",
        "delta_common_zipf",
        "delta_reading_ease",
        "delta_curiosity_style",
    ]
    candidates = metadata["feature"].tolist()
    rows = []
    budget_rows = []
    for feature in candidates:
        controls = _eligible_controls(feature, core_controls, metadata, target)
        x_columns = [feature, *controls]
        beta, hessian, residual = fit_ols(target, Y_COL, x_columns)
        x = design_matrix(target, x_columns)
        weights = if_weight(x, hessian, target_position=1)
        if_resid = weights * residual
        zero_if = weights * y
        tfidf_if = weights * (y - target_pred)
        ols_ifvar = float(np.var(if_resid, ddof=0))
        zero_ifvar = float(np.var(zero_if, ddof=0))
        tfidf_ifvar = float(np.var(tfidf_if, ddof=0))
        raw_col = metadata.set_index("feature").loc[feature, "raw_column"]
        raw = target[raw_col].astype(float)
        abs_weights = np.abs(weights)
        nonzero_share = float((np.abs(raw) > 1e-12).mean())
        unique_values = int(raw.nunique(dropna=True))
        direct_ci_rows = {
            f"direct_ci_halfwidth_B{budget}": 1.96 * math.sqrt(max(ols_ifvar, 0.0) / budget)
            for budget in budgets
        }
        for budget in budgets:
            # This is only a cheap feasibility proxy: use one h_scale-trained TF-IDF surrogate
            # and ask whether the target IF variance reduction would offset labels spent on training.
            for s in [50, 100, 250, 500, 750, 1000, 1500]:
                if s >= budget:
                    continue
                ratio = (tfidf_ifvar / (budget - s)) / (zero_ifvar / budget) if zero_ifvar > 0 else np.nan
                budget_rows.append(
                    {
                        "feature": feature,
                        "budget": int(budget),
                        "s_proxy": int(s),
                        "tfidf_variance_ratio_vs_direct_zero": float(ratio),
                        "passes_proxy": bool(np.isfinite(ratio) and ratio < 1.0),
                    }
                )

        row = {
            "feature": feature,
            "family": metadata.set_index("feature").loc[feature, "family"],
            "kind": metadata.set_index("feature").loc[feature, "kind"],
            "priority": int(metadata.set_index("feature").loc[feature, "priority"]),
            "note": metadata.set_index("feature").loc[feature, "note"],
            "controls": ",".join(controls),
            "beta_controlled": float(beta[1]),
            "abs_beta_controlled": float(abs(beta[1])),
            "nonzero_share": nonzero_share,
            "unique_values": unique_values,
            "sd_raw": float(raw.std(ddof=0)),
            "target_vif": vif_for_target(target, feature, controls),
            "max_abs_corr_with_controls": _safe_abs_corr(target, feature, controls),
            "hessian_condition": float(np.linalg.cond(hessian)),
            "ols_ifvar": ols_ifvar,
            "zero_surrogate_ifvar": zero_ifvar,
            "tfidf_ifvar": tfidf_ifvar,
            "tfidf_ifvar_ratio_vs_zero": float(tfidf_ifvar / zero_ifvar) if zero_ifvar > 0 else np.nan,
            "tfidf_ifvar_ratio_vs_ols": float(tfidf_ifvar / ols_ifvar) if ols_ifvar > 0 else np.nan,
            "if_weight_p50_abs": float(np.quantile(abs_weights, 0.50)),
            "if_weight_p90_abs": float(np.quantile(abs_weights, 0.90)),
            "if_weight_p95_abs": float(np.quantile(abs_weights, 0.95)),
            "if_weight_p99_abs": float(np.quantile(abs_weights, 0.99)),
            "if_weight_max_abs": float(np.max(abs_weights)),
            "if_ess_share": ess_share(weights),
            **direct_ci_rows,
        }
        rows.append(row)

    diagnostics = pd.DataFrame(rows)
    budget = pd.DataFrame(budget_rows)
    diagnostics["passes_nonzero"] = diagnostics["nonzero_share"] >= 0.50
    diagnostics["passes_if_weights"] = (diagnostics["if_weight_p99_abs"] <= 10.0) & (diagnostics["if_ess_share"] >= 0.15)
    diagnostics["passes_collinearity"] = (diagnostics["target_vif"] <= 5.0) & (diagnostics["hessian_condition"] <= 25.0)
    diagnostics["passes_coef"] = diagnostics["abs_beta_controlled"] >= 0.02
    diagnostics["passes_tfidf_proxy"] = diagnostics["tfidf_ifvar_ratio_vs_zero"] < 0.95
    diagnostics["screen_score"] = (
        diagnostics["passes_nonzero"].astype(int) * 20
        + diagnostics["passes_if_weights"].astype(int) * 20
        + diagnostics["passes_collinearity"].astype(int) * 20
        + diagnostics["passes_coef"].astype(int) * 15
        + diagnostics["passes_tfidf_proxy"].astype(int) * 15
        + (4 - diagnostics["priority"].clip(0, 4)) * 2
        + diagnostics["abs_beta_controlled"].clip(0, 0.2) * 10
        - diagnostics["tfidf_ifvar_ratio_vs_zero"].clip(0, 2) * 2
    )
    diagnostics = diagnostics.sort_values(
        ["screen_score", "passes_tfidf_proxy", "abs_beta_controlled", "if_weight_p99_abs"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return diagnostics, budget, tfidf_meta


def write_report(
    output_dir: Path,
    *,
    diagnostics: pd.DataFrame,
    budget: pd.DataFrame,
    metadata: pd.DataFrame,
    tfidf_meta: dict[str, float],
) -> None:
    shortlist_cols = [
        "feature",
        "family",
        "kind",
        "beta_controlled",
        "nonzero_share",
        "target_vif",
        "if_weight_p99_abs",
        "if_ess_share",
        "tfidf_ifvar_ratio_vs_zero",
        "screen_score",
        "note",
    ]
    top = diagnostics.head(20)[shortlist_cols]
    proxy = (
        budget.sort_values("tfidf_variance_ratio_vs_direct_zero")
        .groupby(["feature", "budget"], as_index=False)
        .head(1)
        .sort_values(["budget", "tfidf_variance_ratio_vs_direct_zero"])
        .groupby("budget", as_index=False)
        .head(10)
    )
    lines = [
        "# Upworthy Candidate Feature Screening",
        "",
        "This is a fast target-selection diagnostic. It does not replace Qwen/LoRA evidence.",
        "",
        "## TF-IDF Surrogate",
        "",
        json.dumps(tfidf_meta, indent=2, sort_keys=True),
        "",
        "## Top Candidate Coefficients",
        "",
        dataframe_to_markdown(top, index=False, floatfmt=".4f"),
        "",
        "## Best Cheap Proxy Budget Rows",
        "",
        dataframe_to_markdown(proxy.head(40), index=False, floatfmt=".4f") if not proxy.empty else "(none)",
        "",
        "## Feature Families",
        "",
        dataframe_to_markdown(
            metadata.groupby(["family", "kind"], as_index=False).size().sort_values(["family", "kind"]),
            index=False,
        ),
        "",
    ]
    (output_dir / "screening_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen Upworthy headline features for coefficient-targeted FT+PPI.")
    parser.add_argument("--input-csv", type=Path, default=Path("Data/upworthy_pairs_with_text_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-tfidf-features", type=int, default=50000)
    parser.add_argument("--budgets", default=",".join(str(item) for item in DEFAULT_CI_BUDGETS))
    return parser.parse_args()


def _parse_ints(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(item) for item in value]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input_csv)
    candidate_frame, metadata = build_candidate_features(frame)
    diagnostics, budget, tfidf_meta = summarize_candidates(
        candidate_frame,
        metadata,
        budgets=_parse_ints(args.budgets),
        max_features=args.max_tfidf_features,
    )
    candidate_frame.to_csv(output_dir / "upworthy_pairs_candidate_features.csv", index=False)
    metadata.to_csv(output_dir / "candidate_feature_metadata.csv", index=False)
    diagnostics.to_csv(output_dir / "candidate_diagnostics.csv", index=False)
    budget.to_csv(output_dir / "cheap_tfidf_budget_proxy.csv", index=False)
    candidate_cols = metadata["feature"].tolist()
    candidate_frame.loc[candidate_frame["split"] == "target", candidate_cols].corr().to_csv(
        output_dir / "candidate_correlations_target.csv"
    )
    (output_dir / "tfidf_surrogate_metadata.json").write_text(json.dumps(tfidf_meta, indent=2, sort_keys=True) + "\n")
    write_report(output_dir, diagnostics=diagnostics, budget=budget, metadata=metadata, tfidf_meta=tfidf_meta)
    print(f"wrote {output_dir}")
    print(dataframe_to_markdown(diagnostics.head(15), index=False, floatfmt=".4f"))


if __name__ == "__main__":
    main()
