from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.upworthy_feature_screening import (
    DEFAULT_ALPHAS,
    DEFAULT_BUDGETS,
    DEFAULT_S_GRID,
    SURROGATE_FEATURIZERS,
    feature_name,
    feature_note,
    feature_priority,
    feature_source,
    fit_ols_stats,
    fit_surrogate_featurizer,
    make_screening_split,
    surrogate_matrix,
    train_ridge_select_alpha,
)
from src.experiments.upworthy_question_scaling_law import Y_COL, load_upworthy_pairs


DEFAULT_TARGETS = [
    "delta_QUESTION",
    "delta_NUMERIC",
    "delta_LENGTH",
    "delta_VADER_COMPOUND",
    "delta_HAS_QUOTES",
    "delta_SIMPLICITY",
    "delta_CAPS_SHARE",
    "delta_WE_PRONOUN_SHARE",
]
DEFAULT_FINAL_CONTROLS = [
    "delta_QUESTION",
    "delta_NUMERIC",
    "delta_LENGTH",
    "delta_VADER_COMPOUND",
    "delta_HAS_QUOTES",
    "delta_SIMPLICITY",
]


@dataclass(frozen=True)
class LowDimSpec:
    spec_id: str
    target: str
    controls: tuple[str, ...]
    feature_columns: tuple[str, ...]
    dimension: int
    family: str


def _as_delta(name: str) -> str:
    value = str(name)
    return value if value.startswith("delta_") else f"delta_{value}"


def build_lowdim_specs(
    targets: list[str],
    control_pool: list[str],
    final_controls: list[str],
    dimensions: list[int],
) -> list[LowDimSpec]:
    specs: list[LowDimSpec] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for target in targets:
        allowed_controls = [col for col in control_pool if col != target]
        for dim in sorted(set(dimensions)):
            if dim < 1:
                raise ValueError("dimensions must be positive")
            if dim == 1:
                combos = [()]
                family = "target_only"
            else:
                combos = itertools.combinations(allowed_controls, dim - 1)
                family = f"target_plus_{dim - 1}"
            for controls in combos:
                key = (target, tuple(controls), family)
                if key in seen:
                    continue
                seen.add(key)
                feature_cols = (target, *controls)
                spec_id = "__".join([feature_name(target), *[feature_name(col) for col in controls]])
                specs.append(
                    LowDimSpec(
                        spec_id=spec_id,
                        target=target,
                        controls=tuple(controls),
                        feature_columns=feature_cols,
                        dimension=len(feature_cols),
                        family=family,
                    )
                )
        final = tuple(col for col in final_controls if col != target)
        if final:
            key = (target, final, "final_controls")
            if key not in seen:
                seen.add(key)
                feature_cols = (target, *final)
                spec_id = "__".join([feature_name(target), *[feature_name(col) for col in final]])
                specs.append(
                    LowDimSpec(
                        spec_id=spec_id,
                        target=target,
                        controls=final,
                        feature_columns=feature_cols,
                        dimension=len(feature_cols),
                        family="final_controls",
                    )
                )
    return specs


def compute_spec_summary(df: pd.DataFrame, specs: list[LowDimSpec], budgets: list[int]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    target_df = df.loc[df["split"].astype(str) == "target"].reset_index(drop=True)
    rows = []
    h_scale = df.loc[df["split"].astype(str) == "h_scale"].reset_index(drop=True)
    weights_by_spec: dict[str, np.ndarray] = {}
    for spec in specs:
        stats = fit_ols_stats(target_df, spec.target, list(spec.feature_columns))
        h_x = np.column_stack(
            [np.ones(len(h_scale), dtype=float), h_scale.loc[:, list(spec.feature_columns)].to_numpy(dtype=float)]
        )
        weights_by_spec[spec.spec_id] = h_x @ stats.hessian_inv_target_row
        nonzero_share = float((df[spec.target].to_numpy(dtype=float) != 0).mean())
        row = {
            "spec_id": spec.spec_id,
            "feature": feature_name(spec.target),
            "target": spec.target,
            "controls": ",".join(spec.controls),
            "feature_columns": ",".join(spec.feature_columns),
            "dimension": spec.dimension,
            "family": spec.family,
            "source": feature_source(spec.target),
            "note": feature_note(spec.target),
            "management_priority": feature_priority(spec.target),
            "nonzero_share": nonzero_share,
            "beta": stats.beta,
            "abs_beta": stats.abs_beta,
            "ifvar": stats.ifvar,
            "hessian_condition": stats.hessian_condition,
            "if_weight_p50": stats.if_weight_p50,
            "if_weight_p90": stats.if_weight_p90,
            "if_weight_p95": stats.if_weight_p95,
            "if_weight_p99": stats.if_weight_p99,
            "if_weight_max": stats.if_weight_max,
            "passes_stability": bool(
                nonzero_share >= 0.15 and stats.if_weight_p99 <= 10.0 and math.isfinite(stats.hessian_condition)
            ),
        }
        for budget in budgets:
            row[f"direct_ci_halfwidth_B{budget}"] = 1.96 * math.sqrt(max(stats.ifvar, 0.0) / budget)
        rows.append(row)
    return pd.DataFrame(rows), weights_by_spec


def compute_lowdim_scaling(
    df: pd.DataFrame,
    specs: list[LowDimSpec],
    weights_by_spec: dict[str, np.ndarray],
    *,
    s_values: list[int],
    replications: int,
    seed: int,
    train_pool_size: int,
    validation_stop_size: int,
    validation_scale_size: int,
    alphas: list[float],
    text_featurizer: str,
    max_features: int,
    min_df: int,
    structured_columns: list[str],
) -> pd.DataFrame:
    h_scale = df.loc[df["split"].astype(str) == "h_scale"].reset_index(drop=True)
    if h_scale.empty:
        raise ValueError("input must include split='h_scale'")
    max_s = max(s_values)
    if train_pool_size < max_s:
        raise ValueError("train_pool_size must be at least max(s_grid)")
    featurizer = fit_surrogate_featurizer(
        h_scale,
        mode=text_featurizer,
        max_features=max_features,
        min_df=min_df,
        structured_columns=structured_columns,
    )
    x_text = surrogate_matrix(featurizer, h_scale)
    y = h_scale[Y_COL].to_numpy(dtype=float)
    global_mean = float(np.mean(y))
    rows = []
    spec_lookup = {spec.spec_id: spec for spec in specs}
    for rep in range(int(replications)):
        rng = np.random.default_rng(seed + rep)
        train_pool, stop_idx, scale_idx = make_screening_split(
            len(h_scale), rng, train_pool_size, validation_stop_size, validation_scale_size
        )
        y_scale = y[scale_idx]
        base_ifvar: dict[str, float] = {}
        for s in s_values:
            if s == 0:
                pred_scale = np.full(len(scale_idx), global_mean, dtype=float)
                selected_alpha = np.nan
                stop_mse = np.nan
            else:
                train_idx = train_pool[:s]
                model, selected_alpha, stop_mse = train_ridge_select_alpha(
                    x_text[train_idx], y[train_idx], x_text[stop_idx], y[stop_idx], alphas
                )
                pred_scale = model.predict(x_text[scale_idx])
            residual = y_scale - pred_scale
            for spec_id, weights in weights_by_spec.items():
                spec = spec_lookup[spec_id]
                if_residual = weights[scale_idx] * residual
                ifvar = float(np.var(if_residual, ddof=1))
                if s == 0:
                    base_ifvar[spec_id] = ifvar
                base = base_ifvar.get(spec_id, np.nan)
                rows.append(
                    {
                        "replication": rep,
                        "s": int(s),
                        "spec_id": spec.spec_id,
                        "feature": feature_name(spec.target),
                        "target": spec.target,
                        "dimension": spec.dimension,
                        "family": spec.family,
                        "ifvar": ifvar,
                        "ifvar_ratio_to_s0": float(ifvar / base) if np.isfinite(base) and base > 0 else np.nan,
                        "selected_alpha": selected_alpha,
                        "stop_mse": stop_mse,
                    }
                )
    return pd.DataFrame(rows)


def build_budget_table(scaling: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    grouped = (
        scaling.groupby(["spec_id", "feature", "target", "dimension", "family", "s"], as_index=False)
        .agg(
            mean_ifvar=("ifvar", "mean"),
            mean_ratio=("ifvar_ratio_to_s0", "mean"),
            median_ratio=("ifvar_ratio_to_s0", "median"),
        )
        .sort_values(["spec_id", "s"])
    )
    rows = []
    for key, feature_rows in grouped.groupby(["spec_id", "feature", "target", "dimension", "family"], sort=False):
        spec_id, feature, target, dimension, family = key
        for budget in budgets:
            eligible = feature_rows.loc[(feature_rows["s"] > 0) & (feature_rows["s"] < budget)].copy()
            eligible["win_threshold"] = (budget - eligible["s"]) / budget
            winners = eligible.loc[eligible["mean_ratio"] < eligible["win_threshold"]].sort_values("s")
            if winners.empty:
                best = eligible.sort_values("mean_ratio").head(1)
                if best.empty:
                    best_s = np.nan
                    best_ratio = np.nan
                    threshold = np.nan
                else:
                    row = best.iloc[0]
                    best_s = int(row["s"])
                    best_ratio = float(row["mean_ratio"])
                    threshold = float(row["win_threshold"])
                rows.append(
                    {
                        "spec_id": spec_id,
                        "feature": feature,
                        "target": target,
                        "dimension": int(dimension),
                        "family": family,
                        "budget": int(budget),
                        "wins": False,
                        "best_s": best_s,
                        "best_mean_ratio": best_ratio,
                        "win_threshold_at_best_s": threshold,
                    }
                )
            else:
                row = winners.iloc[0]
                rows.append(
                    {
                        "spec_id": spec_id,
                        "feature": feature,
                        "target": target,
                        "dimension": int(dimension),
                        "family": family,
                        "budget": int(budget),
                        "wins": True,
                        "best_s": int(row["s"]),
                        "best_mean_ratio": float(row["mean_ratio"]),
                        "win_threshold_at_best_s": float(row["win_threshold"]),
                    }
                )
    return pd.DataFrame(rows)


def rank_lowdim_specs(summary: pd.DataFrame, scaling: pd.DataFrame, budget_table: pd.DataFrame) -> pd.DataFrame:
    best_ratio = (
        scaling.loc[scaling["s"] > 0]
        .groupby(["spec_id", "s"], as_index=False)
        .agg(mean_ratio=("ifvar_ratio_to_s0", "mean"), median_ratio=("ifvar_ratio_to_s0", "median"))
        .sort_values(["spec_id", "mean_ratio"])
        .groupby("spec_id", as_index=False)
        .first()
        .rename(columns={"s": "best_s", "mean_ratio": "best_mean_ratio", "median_ratio": "best_median_ratio"})
    )
    wins = budget_table.loc[budget_table["wins"]].sort_values(["spec_id", "budget", "best_s"])
    first_win = wins.groupby("spec_id", as_index=False).first()
    ranked = summary.merge(best_ratio, on="spec_id", how="left").merge(
        first_win[["spec_id", "budget", "best_s", "best_mean_ratio", "win_threshold_at_best_s"]].rename(
            columns={
                "budget": "smallest_winning_budget",
                "best_s": "best_s_at_smallest_winning_budget",
                "best_mean_ratio": "best_ratio_at_smallest_winning_budget",
                "win_threshold_at_best_s": "threshold_at_smallest_winning_budget",
            }
        ),
        on="spec_id",
        how="left",
    )
    ranked["smallest_winning_budget_sort"] = ranked["smallest_winning_budget"].fillna(np.inf)
    ranked = ranked.sort_values(
        [
            "passes_stability",
            "smallest_winning_budget_sort",
            "management_priority",
            "abs_beta",
            "best_mean_ratio",
            "dimension",
            "if_weight_p99",
        ],
        ascending=[False, True, True, False, True, True, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked.drop(columns=["smallest_winning_budget_sort"])


def write_report(output_dir: Path, ranked: pd.DataFrame, budget_table: pd.DataFrame, args: argparse.Namespace) -> None:
    top = ranked.head(20)[
        [
            "rank",
            "feature",
            "dimension",
            "family",
            "controls",
            "beta",
            "abs_beta",
            "if_weight_p99",
            "best_mean_ratio",
            "smallest_winning_budget",
        ]
    ]
    lines = [
        "# Upworthy Low-Dimensional Feature Screening",
        "",
        "This screen varies the target-regression dimension while keeping the same text-derived surrogate.",
        "",
        f"- Input: `{args.input_csv}`",
        f"- Surrogate featurizer: `{args.text_featurizer}`",
        f"- Replications: {args.replications}",
        f"- s grid: {args.s_grid}",
        f"- budgets: {args.budgets}",
        "",
        "## Top Specs",
        "",
        top.to_markdown(index=False),
        "",
        "## Budget Win Counts",
        "",
        budget_table.groupby(["budget", "dimension"])["wins"].sum().reset_index(name="n_winning_specs").to_markdown(index=False),
        "",
    ]
    (output_dir / "lowdim_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-dimensional Upworthy target-regression feature screening.")
    parser.add_argument("--input-csv", type=Path, default=Path("Data/upworthy_pairs_with_text_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/upworthy_m_estimation/lowdim_feature_screening"))
    parser.add_argument("--candidate-features", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--control-pool", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--final-controls", nargs="+", default=DEFAULT_FINAL_CONTROLS)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--s-grid", type=int, nargs="+", default=DEFAULT_S_GRID)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--train-pool-size", type=int, default=5000)
    parser.add_argument("--validation-stop-size", type=int, default=1000)
    parser.add_argument("--validation-scale-size", type=int, default=1000)
    parser.add_argument("--text-featurizer", choices=sorted(SURROGATE_FEATURIZERS), default="word_char_structured")
    parser.add_argument("--tfidf-max-features", type=int, default=50000)
    parser.add_argument("--tfidf-min-df", type=int, default=3)
    parser.add_argument("--outcome-column", default=Y_COL)
    parser.add_argument("--outcome-transform", default="")
    parser.add_argument("--ctr-shrinkage-tau", type=float, default=0.0)
    parser.add_argument("--min-impressions-per-arm", type=float, default=None)
    parser.add_argument("--min-total-impressions", type=float, default=None)
    parser.add_argument("--min-clicks-per-arm", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = [_as_delta(col) for col in args.candidate_features]
    control_pool = [_as_delta(col) for col in args.control_pool]
    final_controls = [_as_delta(col) for col in args.final_controls]
    load_config = {
        "feature_columns": sorted(set(targets) | set(control_pool) | set(final_controls)),
        "outcome_column": args.outcome_column,
        "outcome_transform": args.outcome_transform,
        "ctr_shrinkage_tau": args.ctr_shrinkage_tau,
        "min_impressions_per_arm": args.min_impressions_per_arm,
        "min_total_impressions": args.min_total_impressions,
        "min_clicks_per_arm": args.min_clicks_per_arm,
    }
    df = load_upworthy_pairs(args.input_csv, load_config)
    missing = sorted((set(targets) | set(control_pool) | set(final_controls)) - set(df.columns))
    if missing:
        raise ValueError(f"requested features are missing: {missing}")
    specs = build_lowdim_specs(targets, control_pool, final_controls, list(args.dimensions))
    summary, weights_by_spec = compute_spec_summary(df, specs, list(args.budgets))
    scaling = compute_lowdim_scaling(
        df,
        specs,
        weights_by_spec,
        s_values=sorted(set(int(s) for s in args.s_grid)),
        replications=int(args.replications),
        seed=int(args.seed),
        train_pool_size=int(args.train_pool_size),
        validation_stop_size=int(args.validation_stop_size),
        validation_scale_size=int(args.validation_scale_size),
        alphas=[float(alpha) for alpha in args.alphas],
        text_featurizer=str(args.text_featurizer),
        max_features=int(args.tfidf_max_features),
        min_df=int(args.tfidf_min_df),
        structured_columns=sorted(set(targets) | set(control_pool) | set(final_controls)),
    )
    budget_table = build_budget_table(scaling, list(args.budgets))
    ranked = rank_lowdim_specs(summary, scaling, budget_table)
    summary.to_csv(args.output_dir / "lowdim_spec_summary.csv", index=False)
    scaling.to_csv(args.output_dir / "lowdim_scaling_by_spec_s.csv", index=False)
    budget_table.to_csv(args.output_dir / "lowdim_budget_win_table.csv", index=False)
    ranked.to_csv(args.output_dir / "lowdim_ranked_shortlist.csv", index=False)
    (args.output_dir / "lowdim_args.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir, ranked, budget_table, args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "n_specs": len(specs),
                "top_specs": ranked.head(6)["spec_id"].tolist(),
                "budget_wins": int(budget_table["wins"].sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
