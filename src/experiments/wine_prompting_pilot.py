from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.wine_var_scaling_law import (
    TEXT_COL,
    Y_COL,
    build_population_split,
    clean_wine,
    load_config,
    subset_by_ids,
)


DEFAULT_BUDGETS = [300, 500, 700, 1000]
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_OUTPUT_DIR = Path("artifacts/prompting_pilot_gpt55_n200")

SYSTEM_PROMPT = """I will provide a wine review.
Rate the described wine on a scale of 80--100 based only on the wine review.
Return only valid JSON in the form {"rating": 87}. Do not include any other text."""

FEW_SHOT_EXAMPLES = """Here are two examples of how to rate:

Review:
This is a walk backward after the impressive 2012. Almost impenetrably black, the flavors converge around espresso and bitter chocolate, yet the tannins have a green edge. The wine simply feels flat in the mouth with no life to it.
Rating: 85

Review:
This is one of the great Rieslings from the Wachau, a wonderful panoply of ripe, tropical fruit, pierced with flint, spice and minerality. It is rich and opulent, while never losing sight of the core tautness of a fine Riesling.
Rating: 95"""


def sanitize_error_message(message: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_\-*]+", "[REDACTED_API_KEY]", str(message))


@dataclass(frozen=True)
class VarianceComponents:
    n_valid: int
    y_var: float
    pred_var: float
    residual_var: float
    covariance_y_pred: float
    correlation_y_pred: float
    residual_mean: float
    mae: float
    rmse: float


def build_user_prompt(review: str) -> str:
    return f"""{FEW_SHOT_EXAMPLES}

Now, please rate the following review:

{str(review).strip()}"""


def parse_rating(text: str) -> float | None:
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "rating" in parsed:
            rating = float(parsed["rating"])
            return rating if 80.0 <= rating <= 100.0 else None
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    patterns = [
        r'"rating"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        r"\brating\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"\b([89][0-9]|100)(?:\.[0-9]+)?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            rating = float(match.group(1))
            return rating if 80.0 <= rating <= 100.0 else None
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_jsonl_by_sample_id(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["sample_id"])] = row
    return rows


def build_prompting_sample(config: dict[str, Any], n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = clean_wine(config["input_csv"])
    population = build_population_split(clean, config)
    target = subset_by_ids(clean, population.p_target_ids, "p_target")
    if n > len(target):
        raise ValueError(f"pilot n={n} exceeds target population size={len(target)}")
    rng = np.random.default_rng(int(seed))
    sample_ids = rng.choice(target["sample_id"].to_numpy(dtype=np.int64), size=int(n), replace=False)
    sample = subset_by_ids(clean, sample_ids, "prompting_pilot")
    return target, sample


def ensure_sample(config: dict[str, Any], output_dir: Path, n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_path = output_dir / "prompting_pilot_sample.csv"
    target_path = output_dir / "target_population_ids.csv"
    target, sample = build_prompting_sample(config, n=n, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        if len(sample) != int(n):
            raise ValueError(
                f"existing pilot sample has {len(sample)} rows, but --n={n}; "
                "use a new output directory or remove the old sample file"
            )
    else:
        sample[[ "sample_id", TEXT_COL, Y_COL, "y_raw" ]].to_csv(sample_path, index=False)
    if not target_path.exists():
        target[["sample_id"]].to_csv(target_path, index=False)
    return target, sample


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    output = getattr(response, "output", None)
    if output is not None:
        parts: list[str] = []
        for item in output:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    parts.append(str(text))
        if parts:
            return "\n".join(parts)
    return ""


def call_openai_rating(client: Any, model: str, review: str, max_output_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(review)},
    ]
    response_attempts = [
        {
            "model": model,
            "input": messages,
            "reasoning": {"effort": "minimal"},
            "max_output_tokens": max_output_tokens,
        },
        {
            "model": model,
            "input": messages,
            "max_output_tokens": max_output_tokens,
        },
    ]
    first_error: Exception | None = None
    for kwargs in response_attempts:
        try:
            response = client.responses.create(**kwargs)
            text = _extract_response_text(response)
            if text.strip():
                return text
        except Exception as exc:
            if first_error is None:
                first_error = exc

    chat_attempts = [
        {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_output_tokens,
        },
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_output_tokens,
        },
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_output_tokens,
        },
    ]
    for kwargs in chat_attempts:
        try:
            response = client.chat.completions.create(**kwargs)
            text = str(response.choices[0].message.content or "")
            if text.strip():
                return text
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error
    try:
        raise RuntimeError("OpenAI call failed without returning an exception")
    except RuntimeError:
        raise


def run_api_predictions(
    sample: pd.DataFrame,
    output_dir: Path,
    model: str,
    api_key_env: str,
    max_output_tokens: int,
    sleep_seconds: float,
) -> Path:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{api_key_env} is not set. Set it in the current shell before running API predictions; "
            "the key is never read from files or written to outputs."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("OpenAI API calls require the 'openai' Python package.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "prompting_predictions_raw.jsonl"
    cached = _load_jsonl_by_sample_id(raw_path)
    client = OpenAI(api_key=api_key)
    for _, row in sample.iterrows():
        sample_id = int(row["sample_id"])
        if sample_id in cached and cached[sample_id].get("parse_ok", False):
            continue
        try:
            text = call_openai_rating(client, model=model, review=str(row[TEXT_COL]), max_output_tokens=max_output_tokens)
        except Exception as exc:
            exc_type = type(exc).__name__
            message = sanitize_error_message(str(exc))
            if "Authentication" in exc_type or "invalid_api_key" in message or "Incorrect API key" in message:
                raise RuntimeError(
                    "OpenAI authentication failed. The provided API key was rejected; "
                    "use a current key in OPENAI_API_KEY and rerun."
                ) from None
            raise RuntimeError(
                f"OpenAI request failed for sample_id={sample_id} with {exc_type}: {message}"
            ) from None
        try:
            rating = parse_rating(text)
        except Exception as exc:
            raise RuntimeError(f"failed to parse model response for sample_id={sample_id}") from exc
        _append_jsonl(
            raw_path,
            {
                "sample_id": sample_id,
                "model": model,
                "response_text": text,
                "parsed_rating": rating,
                "parse_ok": rating is not None,
                "true_points": float(row[Y_COL]),
            },
        )
        cached[sample_id] = {"parse_ok": rating is not None}
        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))
    return raw_path


def merge_predictions(sample: pd.DataFrame, raw_path: Path) -> pd.DataFrame:
    rows = list(_load_jsonl_by_sample_id(raw_path).values())
    if not rows:
        raise ValueError(f"no predictions found at {raw_path}")
    pred = pd.DataFrame(rows)
    merged = sample.merge(pred, on="sample_id", how="left", validate="one_to_one")
    merged["pred_raw"] = pd.to_numeric(merged["parsed_rating"], errors="coerce")
    merged["y_raw"] = pd.to_numeric(merged.get("y_raw", merged[Y_COL]), errors="coerce")
    merged["residual_raw"] = merged["y_raw"] - merged["pred_raw"]
    merged["abs_error_raw"] = merged["residual_raw"].abs()
    return merged


def variance_components(predictions: pd.DataFrame) -> VarianceComponents:
    valid = predictions.loc[predictions["pred_raw"].notna()].copy()
    if len(valid) < 3:
        raise ValueError("at least three valid predictions are required")
    y = valid["y_raw"].to_numpy(dtype=float)
    f = valid["pred_raw"].to_numpy(dtype=float)
    residual = y - f
    pred_var = float(np.var(f, ddof=1))
    covariance = float(np.cov(y, f, ddof=1)[0, 1])
    corr = float(covariance / math.sqrt(np.var(y, ddof=1) * pred_var)) if pred_var > 0 else float("nan")
    return VarianceComponents(
        n_valid=int(len(valid)),
        y_var=float(np.var(y, ddof=1)),
        pred_var=pred_var,
        residual_var=float(np.var(residual, ddof=1)),
        covariance_y_pred=covariance,
        correlation_y_pred=corr,
        residual_mean=float(np.mean(residual)),
        mae=float(np.mean(np.abs(residual))),
        rmse=float(math.sqrt(np.mean(residual**2))),
    )


def ppi_variance(residual_var: float, pred_var: float, correction_n: int, unlabeled_n: int) -> float:
    return float(residual_var / correction_n + pred_var / unlabeled_n)


def ppiplus_lambda(covariance_y_pred: float, pred_var: float, correction_n: int, unlabeled_n: int) -> float:
    if pred_var <= 1e-12:
        return 0.0
    return float(covariance_y_pred / pred_var / (1.0 + correction_n / unlabeled_n))


def ppiplus_variance(y: np.ndarray, f: np.ndarray, correction_n: int, unlabeled_n: int) -> tuple[float, float, float]:
    pred_var = float(np.var(f, ddof=1))
    covariance = float(np.cov(y, f, ddof=1)[0, 1]) if pred_var > 0 else 0.0
    lam = ppiplus_lambda(covariance, pred_var, correction_n, unlabeled_n)
    adjusted_residual = y - lam * f
    adjusted_var = float(np.var(adjusted_residual, ddof=1))
    variance = adjusted_var / correction_n + (lam**2) * pred_var / unlabeled_n
    return float(variance), float(lam), float(adjusted_var)


def projection_table(
    target: pd.DataFrame,
    predictions: pd.DataFrame,
    budgets: list[int],
    validation_fraction: float,
    bootstrap_reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = predictions.loc[predictions["pred_raw"].notna()].copy()
    y = valid["y_raw"].to_numpy(dtype=float)
    f = valid["pred_raw"].to_numpy(dtype=float)
    residual = y - f
    target_y = target[Y_COL].to_numpy(dtype=float)
    target_size = int(len(target_y))
    target_var = float(np.var(target_y, ddof=1))
    pred_var = float(np.var(f, ddof=1))
    residual_var = float(np.var(residual, ddof=1))

    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, float | int | str]] = []
    boot_rows: list[dict[str, float | int | str]] = []
    for budget in budgets:
        correction_n = int(budget)
        unlabeled_n = max(2, target_size - int(budget))
        sample_mean_var = target_var / int(budget)
        ppi_var = ppi_variance(residual_var, pred_var, correction_n, unlabeled_n)
        ppip_var, lam, ppip_resid_var = ppiplus_variance(y, f, correction_n, unlabeled_n)
        method_values = {
            "PPI-Only": (ppi_var, 1.0, residual_var),
            "PPI++-Only": (ppip_var, lam, ppip_resid_var),
        }
        for method, (method_var, lam_value, method_resid_var) in method_values.items():
            rows.append(
                {
                    "budget_B": int(budget),
                    "method": method,
                    "pilot_n_valid": int(len(valid)),
                    "correction_n": int(correction_n),
                    "unlabeled_n": int(unlabeled_n),
                    "target_var_y": target_var,
                    "pilot_pred_var": pred_var,
                    "pilot_residual_var": method_resid_var,
                    "lambda": float(lam_value),
                    "projected_var": float(method_var),
                    "projected_rmse": float(math.sqrt(max(method_var, 0.0))),
                    "projected_mae_normal": float(math.sqrt(2.0 / math.pi) * math.sqrt(max(method_var, 0.0))),
                    "projected_variance_reduction": float(1.0 - method_var / sample_mean_var),
                    "projected_sample_savings": float(1.0 - method_var / sample_mean_var),
                }
            )

        if bootstrap_reps > 0:
            n = len(valid)
            for boot_id in range(int(bootstrap_reps)):
                idx = rng.integers(0, n, size=n)
                yb = y[idx]
                fb = f[idx]
                rb = yb - fb
                pred_var_b = float(np.var(fb, ddof=1))
                resid_var_b = float(np.var(rb, ddof=1))
                ppi_var_b = ppi_variance(resid_var_b, pred_var_b, correction_n, unlabeled_n)
                ppip_var_b, lam_b, ppip_resid_var_b = ppiplus_variance(yb, fb, correction_n, unlabeled_n)
                for method, method_var, lam_value, method_resid_var in [
                    ("PPI-Only", ppi_var_b, 1.0, resid_var_b),
                    ("PPI++-Only", ppip_var_b, lam_b, ppip_resid_var_b),
                ]:
                    boot_rows.append(
                        {
                            "budget_B": int(budget),
                            "method": method,
                            "bootstrap_id": int(boot_id),
                            "lambda": float(lam_value),
                            "pilot_residual_var": float(method_resid_var),
                            "projected_var": float(method_var),
                            "projected_rmse": float(math.sqrt(max(method_var, 0.0))),
                            "projected_mae_normal": float(
                                math.sqrt(2.0 / math.pi) * math.sqrt(max(method_var, 0.0))
                            ),
                            "projected_variance_reduction": float(1.0 - method_var / sample_mean_var),
                            "projected_sample_savings": float(1.0 - method_var / sample_mean_var),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(boot_rows)


def summarize_bootstrap(point: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    if boot.empty:
        point = point.copy()
        for col in [
            "projected_var",
            "projected_rmse",
            "projected_mae_normal",
            "projected_variance_reduction",
            "projected_sample_savings",
        ]:
            point[f"{col}_se"] = np.nan
            point[f"{col}_ci_low"] = np.nan
            point[f"{col}_ci_high"] = np.nan
        return point
    summaries = []
    metrics = [
        "lambda",
        "pilot_residual_var",
        "projected_var",
        "projected_rmse",
        "projected_mae_normal",
        "projected_variance_reduction",
        "projected_sample_savings",
    ]
    for keys, group in boot.groupby(["budget_B", "method"], sort=True):
        row: dict[str, Any] = {"budget_B": keys[0], "method": keys[1]}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_se"] = float(np.std(values, ddof=1))
            row[f"{metric}_ci_low"] = float(np.quantile(values, 0.025))
            row[f"{metric}_ci_high"] = float(np.quantile(values, 0.975))
        summaries.append(row)
    return point.merge(pd.DataFrame(summaries), on=["budget_B", "method"], how="left")


def _fmt_est(value: float, se: float | None, digits: int = 3) -> str:
    if not np.isfinite(value):
        return r"\NA"
    if se is None or not np.isfinite(se):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ({se:.{digits}f})"


def write_latex_rows(projection: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "% Auto-generated by src/experiments/wine_prompting_pilot.py.",
        "% Values are variance-based projections from the prompting pilot, not empirical repeated-run errors.",
    ]
    method_order = {"PPI-Only": 0, "PPI++-Only": 1}
    for budget, group in projection.groupby("budget_B", sort=True):
        group = group.copy()
        group["_method_order"] = group["method"].map(method_order).fillna(99)
        for row_index, (_, row) in enumerate(group.sort_values(["_method_order", "method"]).iterrows()):
            method = str(row["method"])
            rmse = _fmt_est(float(row["projected_rmse"]), float(row.get("projected_rmse_se", np.nan)))
            mae = _fmt_est(float(row["projected_mae_normal"]), float(row.get("projected_mae_normal_se", np.nan)))
            prefix = f"{int(budget)}" if row_index == 0 else "    "
            lines.append(
                f"{prefix} & {method} & GPT-5.5 pilot projection & {rmse} & {mae} & \\NA \\\\"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_predictions(
    target: pd.DataFrame,
    sample: pd.DataFrame,
    raw_path: Path,
    output_dir: Path,
    budgets: list[int],
    validation_fraction: float,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    predictions = merge_predictions(sample, raw_path)
    predictions.to_csv(output_dir / "prompting_predictions.csv", index=False)
    components = variance_components(predictions)
    point, boot = projection_table(
        target=target,
        predictions=predictions,
        budgets=budgets,
        validation_fraction=validation_fraction,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )
    projection = summarize_bootstrap(point, boot)
    point.to_csv(output_dir / "ppi_projection_by_budget.csv", index=False)
    if not boot.empty:
        boot.to_csv(output_dir / "ppi_projection_bootstrap.csv", index=False)
    projection.to_csv(output_dir / "ppi_projection_summary.csv", index=False)
    write_latex_rows(projection, output_dir / "ppi_projection_latex_rows.tex")

    summary = {
        "pilot_n_valid": components.n_valid,
        "pilot_y_var": components.y_var,
        "pilot_prediction_var": components.pred_var,
        "pilot_residual_var": components.residual_var,
        "pilot_covariance_y_prediction": components.covariance_y_pred,
        "pilot_correlation_y_prediction": components.correlation_y_pred,
        "pilot_residual_mean": components.residual_mean,
        "pilot_mae": components.mae,
        "pilot_rmse": components.rmse,
        "target_size": int(len(target)),
        "target_mean": float(target[Y_COL].mean()),
        "target_var_y": float(target[Y_COL].var(ddof=1)),
        "valid_parse_rate": float(predictions["pred_raw"].notna().mean()),
        "budgets": budgets,
        "projection_note": (
            "Projection uses the 200-example prompting pilot to estimate GPT residual and prediction "
            "variance components. RMSE/MAE are variance-based projections, not empirical repeated-run errors."
        ),
    }
    _write_json(output_dir / "prompting_pilot_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small GPT prompting pilot for Wine PPI/PPI++ baselines.")
    parser.add_argument("--config", default="configs/wine_full_grid_allocation.yaml")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--prepare-only", action="store_true", help="Create the fixed pilot sample without API calls.")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip API calls and analyze an existing prompting_predictions_raw.jsonl file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    target, sample = ensure_sample(config, output_dir=output_dir, n=args.n, seed=args.seed)
    metadata = {
        "config": args.config,
        "model": args.model,
        "n": int(args.n),
        "seed": int(args.seed),
        "budgets": [int(x) for x in args.budgets],
        "api_key_env": args.api_key_env,
        "prompting_note": "The API key is read only from the named environment variable and is not written to outputs.",
    }
    _write_json(output_dir / "prompting_pilot_metadata.json", metadata)

    raw_path = output_dir / "prompting_predictions_raw.jsonl"
    if args.prepare_only:
        print(f"prepared sample: {output_dir / 'prompting_pilot_sample.csv'}")
        return
    if not args.analyze_only:
        raw_path = run_api_predictions(
            sample=sample,
            output_dir=output_dir,
            model=args.model,
            api_key_env=args.api_key_env,
            max_output_tokens=args.max_output_tokens,
            sleep_seconds=args.sleep_seconds,
        )
    summary = analyze_predictions(
        target=target,
        sample=sample,
        raw_path=raw_path,
        output_dir=output_dir,
        budgets=[int(x) for x in args.budgets],
        validation_fraction=float(config.get("validation_fraction", 0.2)),
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 17,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
