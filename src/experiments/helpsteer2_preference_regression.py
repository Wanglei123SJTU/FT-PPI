from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.formatting import dataframe_to_markdown


Y_COL = "y_preference_strength"
DEFAULT_FEATURES = ["delta_log_length", "delta_prompt_coverage", "delta_format"]
DEFAULT_TARGETS = ["delta_log_length", "delta_prompt_coverage", "delta_format"]
DEFAULT_BUDGETS = [500, 1000, 1500, 3000, 5000]

FEATURE_NOTES = {
    "delta_log_length": (
        "AI product/reward-design audit: whether human preference systematically rewards longer responses, "
        "controlling for prompt coverage and structured formatting."
    ),
    "delta_prompt_coverage": (
        "Instruction-following audit: whether responses that lexically cover more of the prompt receive higher "
        "preference, controlling for length and formatting."
    ),
    "delta_format": (
        "Response-design audit: whether structured formatting such as bullets, numbered lists, or markdown receives "
        "higher preference, controlling for length and prompt coverage."
    ),
}


@dataclass(frozen=True)
class OlsResult:
    model: str
    target: str
    feature_columns: list[str]
    beta: float
    intercept: float
    residual_var: float
    ifvar: float
    standard_error_full_population: float
    hessian_condition: float
    if_weight_p50: float
    if_weight_p90: float
    if_weight_p95: float
    if_weight_p99: float
    if_weight_max: float
    nonzero_share: float
    raw_mean: float
    raw_sd: float


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _as_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    x = frame[feature_columns].astype(float).to_numpy()
    return np.column_stack([np.ones(len(frame), dtype=float), x])


def _safe_inverse(matrix: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    if ridge > 0:
        matrix = matrix + ridge * np.eye(matrix.shape[0])
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def fit_ols_result(
    frame: pd.DataFrame,
    *,
    target: str,
    feature_columns: list[str],
    model: str,
    ridge: float = 0.0,
) -> OlsResult:
    if target not in feature_columns:
        raise ValueError(f"target={target} must be included in feature_columns={feature_columns}")
    if Y_COL not in frame.columns:
        raise ValueError(f"missing outcome column: {Y_COL}")
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing feature columns: {missing}")

    clean = frame.dropna(subset=[Y_COL, *feature_columns]).copy()
    if len(clean) <= len(feature_columns) + 1:
        raise ValueError("not enough rows for OLS")

    y = clean[Y_COL].astype(float).to_numpy()
    design = _as_matrix(clean, feature_columns)
    hessian = design.T @ design / len(clean)
    hessian_inv = _safe_inverse(hessian, ridge=ridge)
    beta = hessian_inv @ (design.T @ y / len(clean))
    y_hat = design @ beta
    residual = y - y_hat

    target_index = feature_columns.index(target) + 1
    if_weights = design @ hessian_inv[target_index, :]
    if_residual = if_weights * residual

    target_values = clean[target].astype(float).to_numpy()
    quantiles = np.quantile(np.abs(if_weights), [0.5, 0.9, 0.95, 0.99])
    ifvar = float(np.var(if_residual, ddof=0))
    return OlsResult(
        model=model,
        target=target,
        feature_columns=feature_columns,
        beta=float(beta[target_index]),
        intercept=float(beta[0]),
        residual_var=float(np.var(residual, ddof=0)),
        ifvar=ifvar,
        standard_error_full_population=float(math.sqrt(ifvar / len(clean))),
        hessian_condition=float(np.linalg.cond(hessian)),
        if_weight_p50=float(quantiles[0]),
        if_weight_p90=float(quantiles[1]),
        if_weight_p95=float(quantiles[2]),
        if_weight_p99=float(quantiles[3]),
        if_weight_max=float(np.max(np.abs(if_weights))),
        nonzero_share=float(np.mean(np.abs(target_values) > 1e-12)),
        raw_mean=float(np.mean(target_values)),
        raw_sd=float(np.std(target_values, ddof=0)),
    )


def result_to_dict(result: OlsResult) -> dict:
    payload = {
        "model": result.model,
        "target": result.target,
        "feature_columns": ",".join(result.feature_columns),
        "beta": result.beta,
        "abs_beta": abs(result.beta),
        "intercept": result.intercept,
        "residual_var": result.residual_var,
        "ifvar": result.ifvar,
        "se_full_population": result.standard_error_full_population,
        "ci95_half_width_full_population": 1.96 * result.standard_error_full_population,
        "hessian_condition": result.hessian_condition,
        "if_weight_p50": result.if_weight_p50,
        "if_weight_p90": result.if_weight_p90,
        "if_weight_p95": result.if_weight_p95,
        "if_weight_p99": result.if_weight_p99,
        "if_weight_max": result.if_weight_max,
        "nonzero_share": result.nonzero_share,
        "raw_mean": result.raw_mean,
        "raw_sd": result.raw_sd,
        "management_note": FEATURE_NOTES.get(result.target, ""),
    }
    if result.target == "delta_log_length":
        payload["effect_of_doubling_response_length"] = result.beta * math.log(2.0)
    return payload


def build_regression_summary(
    frame: pd.DataFrame,
    *,
    targets: Iterable[str] = DEFAULT_TARGETS,
    controlled_features: Iterable[str] = DEFAULT_FEATURES,
    ridge: float = 0.0,
) -> pd.DataFrame:
    rows: list[dict] = []
    controlled = list(controlled_features)
    for target in targets:
        rows.append(
            result_to_dict(
                fit_ols_result(
                    frame,
                    target=target,
                    feature_columns=[target],
                    model="marginal",
                    ridge=ridge,
                )
            )
        )
        if target in controlled:
            rows.append(
                result_to_dict(
                    fit_ols_result(
                        frame,
                        target=target,
                        feature_columns=controlled,
                        model="controlled",
                        ridge=ridge,
                    )
                )
            )
    return pd.DataFrame(rows)


def build_budget_ci_proxy(summary: pd.DataFrame, *, budgets: Iterable[int] = DEFAULT_BUDGETS) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in summary.iterrows():
        for budget in budgets:
            se = math.sqrt(float(row["ifvar"]) / budget)
            rows.append(
                {
                    "model": row["model"],
                    "target": row["target"],
                    "budget": int(budget),
                    "se_proxy": se,
                    "ci95_half_width_proxy": 1.96 * se,
                    "ci95_length_proxy": 2.0 * 1.96 * se,
                }
            )
    return pd.DataFrame(rows)


def rank_targets(summary: pd.DataFrame) -> pd.DataFrame:
    controlled = summary.loc[summary["model"] == "controlled"].copy()
    if controlled.empty:
        return controlled
    controlled["stable"] = (
        (controlled["nonzero_share"] >= 0.10)
        & (controlled["if_weight_p99"] <= 10.0)
        & np.isfinite(controlled["hessian_condition"])
        & (controlled["hessian_condition"] <= 1e6)
    )
    controlled["priority"] = controlled["target"].map(
        {"delta_format": 0, "delta_prompt_coverage": 1, "delta_log_length": 2}
    ).fillna(9)
    controlled = controlled.sort_values(
        ["stable", "abs_beta", "if_weight_p99", "priority"],
        ascending=[False, False, True, True],
    )
    return controlled


def write_report(
    path: str | Path,
    *,
    input_csv: str | Path,
    summary: pd.DataFrame,
    budgets: pd.DataFrame,
    ranked: pd.DataFrame,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    controlled = summary.loc[summary["model"] == "controlled"].copy()
    budget_focus = budgets.loc[(budgets["model"] == "controlled") & (budgets["budget"].isin([500, 1000, 3000]))]

    lines = [
        "# HelpSteer2 Preference Pairwise Regression Screen",
        "",
        f"Input CSV: `{input_csv}`",
        "",
        "Outcome sign convention: positive `preference_strength` means response 2 is preferred over response 1.",
        "All covariates are response 2 minus response 1, so positive coefficients mean the higher-valued response tends to be preferred.",
        "",
        "## Controlled Coefficients",
        "",
        dataframe_to_markdown(controlled[
            [
                "target",
                "beta",
                "abs_beta",
                "ifvar",
                "if_weight_p99",
                "nonzero_share",
                "hessian_condition",
            ]
        ], index=False, floatfmt=".4f"),
        "",
        "## Direct Label-Only CI Proxy",
        "",
        dataframe_to_markdown(budget_focus[["target", "budget", "ci95_length_proxy"]], index=False, floatfmt=".4f"),
        "",
        "## Ranked Targets",
        "",
        dataframe_to_markdown(ranked[
            [
                "target",
                "stable",
                "beta",
                "abs_beta",
                "if_weight_p99",
                "nonzero_share",
                "management_note",
            ]
        ], index=False, floatfmt=".4f")
        if not ranked.empty
        else "(no controlled target rows)",
        "",
        "## Interpretation Notes",
        "",
        "- `delta_log_length`: coefficient times log(2) is the preference-strength change associated with doubling response length, controlling for coverage and format.",
        "- `delta_prompt_coverage`: effect of covering more prompt tokens in response 2 than response 1.",
        "- `delta_format`: effect of structured formatting in response 2 relative to response 1.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen HelpSteer2 preference pairwise regression targets.")
    parser.add_argument("--input-csv", type=str, default="Data/helpsteer2_preference_pairs.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts/helpsteer2_preference/regression_screen")
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    parser.add_argument("--budgets", nargs="*", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--ridge", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_regression_summary(frame, targets=args.targets, controlled_features=args.features, ridge=args.ridge)
    budgets = build_budget_ci_proxy(summary, budgets=args.budgets)
    ranked = rank_targets(summary)

    summary_path = output_dir / "regression_summary.csv"
    budgets_path = output_dir / "direct_label_ci_proxy.csv"
    ranked_path = output_dir / "ranked_targets.csv"
    report_path = output_dir / "screening_report.md"
    metadata_path = output_dir / "metadata.json"

    summary.to_csv(summary_path, index=False)
    budgets.to_csv(budgets_path, index=False)
    ranked.to_csv(ranked_path, index=False)
    write_report(report_path, input_csv=args.input_csv, summary=summary, budgets=budgets, ranked=ranked)
    write_json(
        metadata_path,
        {
            "input_csv": args.input_csv,
            "output_dir": str(output_dir),
            "targets": args.targets,
            "features": args.features,
            "budgets": args.budgets,
            "ridge": args.ridge,
            "n_rows": int(len(frame)),
        },
    )

    print(f"wrote {summary_path}")
    print(f"wrote {budgets_path}")
    print(f"wrote {ranked_path}")
    print(f"wrote {report_path}")
    print(ranked[["target", "beta", "ifvar", "if_weight_p99", "nonzero_share"]].to_string(index=False))


if __name__ == "__main__":
    main()
