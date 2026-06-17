from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import t


METHOD_LABELS = {
    "mse_stop_mse": "MSE loss, stop by MSE",
    "mse_stop_var": "MSE loss, stop by Var",
    "var_stop_var": "Var loss, stop by Var",
    "mse": "MSE loss",
    "var": "Var loss",
}

METHOD_COLORS = {
    # Okabe-Ito inspired, colorblind-friendly tones used in many ML papers.
    "mse_stop_mse": "#4477AA",
    "mse_stop_var": "#EE7733",
    "var_stop_var": "#228833",
    "mse": "#EE7733",
    "var": "#228833",
}

METHOD_MARKERS = {
    "mse_stop_mse": "o",
    "mse_stop_var": "s",
    "var_stop_var": "^",
    "mse": "o",
    "var": "o",
}

SCALING_FIT_COLOR = "#E66101"
MEASURED_VAR_COLOR = "#005A9C"
VAR_METHOD_COLOR = "#1B7837"
NOISE_FLOOR_COLOR = "#9E9E9E"
BASELINE_COLOR = "#9C2F6F"


@dataclass(frozen=True)
class ScalingFit:
    method: str
    a: float
    alpha: float
    b: float
    sse: float
    rmse: float
    r2: float
    s_star: float
    objective: float


@dataclass(frozen=True)
class PlotData:
    frame: pd.DataFrame
    has_replications: bool
    baseline_var: float
    value_label: str
    value_stem: str


def power_law(s: np.ndarray, a: float, alpha: float, b: float) -> np.ndarray:
    return float(a) * np.asarray(s, dtype=float) ** (-float(alpha)) + float(b)


def fit_power_law(
    train_sizes: np.ndarray,
    residual_vars: np.ndarray,
    baseline_var: float,
    n_eff: float,
) -> ScalingFit:
    x = np.asarray(train_sizes, dtype=float)
    y = np.asarray(residual_vars, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return ScalingFit("", math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)

    b_upper = max(float(baseline_var), float(np.max(y)), 1e-9)
    lower = np.array([1e-12, 0.02, 0.0], dtype=float)
    upper = np.array([np.inf, 1.5, b_upper], dtype=float)

    starts = []
    for alpha0 in [0.05, 0.1, 0.2, 0.5, 0.9, 1.3]:
        for b0 in [0.0, min(float(np.min(y)) * 0.25, b_upper), min(float(np.min(y)) * 0.75, b_upper)]:
            a0 = max(float(np.max(y) - b0), 1e-6) * float(np.min(x) ** alpha0)
            starts.append(np.array([a0, alpha0, b0], dtype=float))

    best_params: np.ndarray | None = None
    best_sse = math.inf
    for start in starts:
        result = least_squares(
            lambda params: power_law(x, params[0], params[1], params[2]) - y,
            x0=np.maximum(start, lower),
            bounds=(lower, upper),
            max_nfev=20_000,
        )
        sse = float(np.sum(result.fun**2))
        if sse < best_sse:
            best_sse = sse
            best_params = result.x

    if best_params is None:
        return ScalingFit("", math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)

    a, alpha, b = [float(v) for v in best_params]
    pred = power_law(x, a, alpha, b)
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - best_sse / sst) if sst > 0 else math.nan

    def objective(s_value: float) -> float:
        correction = float(n_eff) - float(s_value)
        if correction <= 0:
            return math.inf
        return float(power_law(np.array([s_value]), a, alpha, b)[0] / correction)

    upper_s = max(float(n_eff) - 1.0, 1.0)
    opt = minimize_scalar(objective, bounds=(1.0, upper_s), method="bounded")
    return ScalingFit(
        method="",
        a=a,
        alpha=alpha,
        b=b,
        sse=best_sse,
        rmse=float(np.sqrt(best_sse / len(x))),
        r2=r2,
        s_star=float(opt.x),
        objective=float(opt.fun),
    )


def clean_wine_baseline(data_path: Path) -> float | None:
    if not data_path.exists():
        return None
    df = pd.read_csv(data_path, usecols=["description", "points"])
    df = df.dropna(subset=["description", "points"]).drop_duplicates(subset=["description"], keep="first")
    return float(pd.to_numeric(df["points"], errors="coerce").dropna().var(ddof=1))


def _resolve_value_column(raw: pd.DataFrame, value: str) -> tuple[str, float, str, str]:
    candidates = {
        "validation_scale": [
            ("validation_scale_residual_var_raw", 1.0),
            ("validation_scale_residual_var", 25.0),
            ("validation_residual_var_raw", 1.0),
            ("validation_residual_var", 25.0),
        ],
        "validation_stop": [
            ("validation_stop_residual_var_raw", 1.0),
            ("validation_stop_residual_var", 25.0),
        ],
    }
    labels = {
        "validation_scale": "Validation-scale residual variance",
        "validation_stop": "Validation-stop residual variance",
    }
    stems = {
        "validation_scale": "validation_scale_residual_variance",
        "validation_stop": "validation_stop_residual_variance",
    }
    if value == "auto":
        search_order = ["validation_scale", "validation_stop"]
    else:
        if value not in candidates:
            raise ValueError(f"Unknown --value '{value}'")
        search_order = [value]
    for key in search_order:
        for column, multiplier in candidates[key]:
            if column in raw.columns:
                return column, multiplier, labels[key], stems[key]
    expected = ", ".join(col for pairs in candidates.values() for col, _ in pairs)
    raise ValueError(f"No supported residual-variance column found. Expected one of: {expected}")


def load_results(input_dir: Path, data_path: Path, value: str) -> PlotData:
    runtime_path = input_dir / "training_runtime_summary.csv"
    mean_path = input_dir / "loss_comparison_residual_variance_from_log.csv"
    comparison_path = input_dir / "loss_comparison_summary.csv"

    baseline = clean_wine_baseline(data_path)
    if baseline is None:
        baseline = 0.382673 * 25.0

    if runtime_path.exists():
        raw = pd.read_csv(runtime_path)
        value_col, multiplier, value_label, value_stem = _resolve_value_column(raw, value)
        method_series = raw["loss"].astype(str) if "loss" in raw.columns else pd.Series("var", index=raw.index)
        df = pd.DataFrame(
            {
                "method": method_series,
                "replication_id": raw.get("replication_id", pd.Series(np.nan, index=raw.index)),
                "s_train": raw["s_train"].astype(int),
                "residual_var_raw": raw[value_col].astype(float) * multiplier,
            }
        )
        has_replications = "replication_id" in raw.columns and raw["replication_id"].nunique(dropna=True) > 1
        return PlotData(df, has_replications, baseline, value_label, value_stem)

    if mean_path.exists():
        raw = pd.read_csv(mean_path)
        method_col = "method" if "method" in raw.columns else "loss"
        value_col, multiplier, value_label, value_stem = _resolve_value_column(raw, value)
        df = pd.DataFrame(
            {
                "method": raw[method_col].astype(str),
                "replication_id": np.nan,
                "s_train": raw["s_train"].astype(int),
                "residual_var_raw": raw[value_col].astype(float) * multiplier,
            }
        )
        return PlotData(df, False, baseline, value_label, value_stem)

    if comparison_path.exists():
        raw = pd.read_csv(comparison_path)
        raw = raw[raw["loss"].astype(str).isin(METHOD_LABELS)]
        value_col, multiplier, value_label, value_stem = _resolve_value_column(raw, value)
        df = pd.DataFrame(
            {
                "method": raw["loss"].astype(str),
                "replication_id": np.nan,
                "s_train": raw["s_train"].astype(int),
                "residual_var_raw": raw[value_col].astype(float) * multiplier,
            }
        )
        return PlotData(df, False, baseline, value_label, value_stem)

    raise FileNotFoundError(
        f"No supported result file found in {input_dir}. Expected training_runtime_summary.csv, "
        "loss_comparison_residual_variance_from_log.csv, or loss_comparison_summary.csv."
    )


def summarize(df: pd.DataFrame, has_replications: bool) -> pd.DataFrame:
    rows = []
    for (method, s_train), group in df.groupby(["method", "s_train"], sort=True):
        values = group["residual_var_raw"].to_numpy(dtype=float)
        n = int(np.isfinite(values).sum())
        mean = float(np.nanmean(values))
        if has_replications and n > 1:
            sd = float(np.nanstd(values, ddof=1))
            se = sd / math.sqrt(n)
            half_width = float(t.ppf(0.975, n - 1) * se)
        else:
            sd = math.nan
            se = math.nan
            half_width = 0.0
        rows.append(
            {
                "method": method,
                "s_train": int(s_train),
                "n": n,
                "mean": mean,
                "sd": sd,
                "se": se,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "s_train"]).reset_index(drop=True)


def fit_all(summary: pd.DataFrame, baseline_var: float, n_eff: float) -> pd.DataFrame:
    rows = []
    for method, group in summary.groupby("method", sort=False):
        group = group.sort_values("s_train")
        fit = fit_power_law(group["s_train"].to_numpy(), group["mean"].to_numpy(), baseline_var, n_eff)
        rows.append(
            {
                "method": method,
                "a": fit.a,
                "alpha": fit.alpha,
                "b": fit.b,
                "rmse": fit.rmse,
                "r2": fit.r2,
                "s_star": fit.s_star,
                "objective": fit.objective,
            }
        )
    return pd.DataFrame(rows)


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def method_color(method: str) -> str:
    if method == "mse_stop_mse":
        return MEASURED_VAR_COLOR
    if method == "mse_stop_var":
        return SCALING_FIT_COLOR
    if method == "var_stop_var":
        return VAR_METHOD_COLOR
    return METHOD_COLORS.get(method, "#4C78A8")


def method_marker(method: str) -> str:
    return METHOD_MARKERS.get(method, "o")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.titlesize": 9.4,
            "axes.labelsize": 10.0,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def apply_paper_axis(ax: plt.Axes, *, minor_grid: bool = False) -> None:
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.yaxis.grid(True, which="major", color="#DADADA", lw=0.72, ls="-", alpha=0.82)
    if minor_grid:
        ax.minorticks_on()
        ax.yaxis.grid(True, which="minor", color="#EFEFEF", lw=0.45, ls="-", alpha=0.78)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#2F2F2F")
    ax.tick_params(axis="both", which="major", length=4.0, color="#2F2F2F")
    ax.tick_params(axis="both", which="minor", length=2.4, color="#666666")


def add_top_legend(ax: plt.Axes, *, ncol: int) -> None:
    legend = ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=ncol,
        columnspacing=1.25,
        handlelength=2.2,
        handletextpad=0.55,
        borderaxespad=0.0,
    )


def apply_reference_scaling_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", color="#BDBDBD", lw=0.55, ls=":", alpha=0.9)
    ax.grid(False, which="minor")
    ax.minorticks_off()
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(0.9)
    ax.tick_params(axis="both", which="major", length=3.5, width=0.8, color="black")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def preview_note(ax: plt.Axes, has_replications: bool) -> None:
    if has_replications:
        return
    fig = ax.figure
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.12,
        0.03,
        "Mean-only preview; 95% CI will be drawn automatically when per-replication rows are present.",
        ha="left",
        va="bottom",
        color="#555555",
        fontsize=8.5,
        style="italic",
    )


def plot_linear_scaling(
    summary: pd.DataFrame,
    fits: pd.DataFrame,
    output_dir: Path,
    primary_method: str,
    has_replications: bool,
    value_label: str,
    value_stem: str,
) -> None:
    group = summary[summary["method"] == primary_method].sort_values("s_train")
    fit = fits[fits["method"] == primary_method].iloc[0]
    dense = np.linspace(float(group["s_train"].min()), float(group["s_train"].max()), 500)
    yerr = np.vstack([group["mean"] - group["ci95_low"], group["ci95_high"] - group["mean"]])

    fig, ax = plt.subplots(figsize=(4.9, 3.9))
    ax.plot(dense, power_law(dense, fit["a"], fit["alpha"], fit["b"]), color=SCALING_FIT_COLOR, lw=1.5, label="Scaling fit")
    ax.axhline(fit["b"], color=NOISE_FLOOR_COLOR, lw=1.0, ls="--", label="b (Noise Floor)")
    ax.errorbar(
        group["s_train"],
        group["mean"],
        yerr=yerr if has_replications else None,
        fmt="o",
        ms=4.7,
        color=MEASURED_VAR_COLOR,
        ecolor=MEASURED_VAR_COLOR,
        elinewidth=0.9,
        capsize=2.5,
        mec="black",
        mew=0.6,
        label="Measured Var" if has_replications else "Measured Var (mean only)",
    )
    ax.set_xlabel("FT subset size (s)", fontweight="bold")
    ax.set_ylabel(r"$\mathrm{Var}(Y - f_s(X))$", fontweight="bold")
    ax.set_title("")
    apply_reference_scaling_axis(ax)
    ax.legend(frameon=False, loc="upper right", handlelength=2.0, borderaxespad=0.45, fontsize=7.8)
    fig.tight_layout(pad=0.25)
    preview_note(ax, has_replications)
    save_figure(fig, output_dir, f"{primary_method}_{value_stem}_power_law_linear_raw")


def plot_loglog_scaling(
    summary: pd.DataFrame,
    fits: pd.DataFrame,
    output_dir: Path,
    primary_method: str,
    has_replications: bool,
    value_label: str,
    value_stem: str,
) -> None:
    group = summary[summary["method"] == primary_method].sort_values("s_train").copy()
    fit = fits[fits["method"] == primary_method].iloc[0]
    eps = 1e-12
    centered = np.maximum(group["mean"].to_numpy(dtype=float) - float(fit["b"]), eps)
    xlog = np.log10(group["s_train"].to_numpy(dtype=float))
    ylog = np.log10(centered)

    lower_centered = np.maximum(group["ci95_low"].to_numpy(dtype=float) - float(fit["b"]), eps)
    upper_centered = np.maximum(group["ci95_high"].to_numpy(dtype=float) - float(fit["b"]), eps)
    yerr = np.vstack([ylog - np.log10(lower_centered), np.log10(upper_centered) - ylog])
    dense = np.linspace(float(group["s_train"].min()), float(group["s_train"].max()), 500)

    fig, ax = plt.subplots(figsize=(4.9, 3.9))
    ax.plot(
        np.log10(dense),
        np.log10(np.maximum(power_law(dense, fit["a"], fit["alpha"], fit["b"]) - fit["b"], eps)),
        color=SCALING_FIT_COLOR,
        lw=1.5,
        label=f"Scaling Fit (slope = {-float(fit['alpha']):.3f})",
    )
    ax.errorbar(
        xlog,
        ylog,
        yerr=yerr if has_replications else None,
        fmt="o",
        ms=4.7,
        color=MEASURED_VAR_COLOR,
        ecolor=MEASURED_VAR_COLOR,
        elinewidth=0.9,
        capsize=2.5,
        mec="black",
        mew=0.6,
        label="Measured (Var - b)" if has_replications else "Measured (Var - b, mean only)",
    )
    ax.set_xlabel("log(FT subset size)", fontweight="bold")
    ax.set_ylabel(r"$\log(\mathrm{Var}(Y - f_s(X)) - b)$", fontweight="bold")
    ax.set_title("")
    apply_reference_scaling_axis(ax)
    ax.legend(frameon=False, loc="upper right", handlelength=2.0, borderaxespad=0.45, fontsize=7.8)
    fig.tight_layout(pad=0.25)
    preview_note(ax, has_replications)
    save_figure(fig, output_dir, f"{primary_method}_{value_stem}_power_law_loglog_raw")


def plot_method_comparison(
    summary: pd.DataFrame,
    output_dir: Path,
    baseline_var: float,
    has_replications: bool,
    value_label: str,
    value_stem: str,
) -> None:
    methods = [m for m in ["mse_stop_mse", "mse_stop_var", "var_stop_var", "mse", "var"] if m in set(summary["method"])]
    methods += [m for m in summary["method"].unique() if m not in methods]

    fig, ax = plt.subplots(figsize=(6.75, 4.35))
    for method in methods:
        group = summary[summary["method"] == method].sort_values("s_train")
        color = method_color(method)
        ax.plot(
            group["s_train"],
            group["mean"],
            color=color,
            lw=1.65,
            marker=method_marker(method),
            ms=4.9,
            mec="black",
            mew=0.55,
            label=method_label(method),
        )
        if has_replications:
            ax.fill_between(group["s_train"], group["ci95_low"], group["ci95_high"], color=color, alpha=0.10, lw=0)
    ax.axhline(baseline_var, color=BASELINE_COLOR, lw=1.2, ls="--")
    x_min = float(summary["s_train"].min())
    x_max = float(summary["s_train"].max())
    ax.text(
        x_max - 12,
        baseline_var + 0.02,
        f"Baseline Var = {baseline_var:.3f}",
        ha="right",
        va="bottom",
        color=BASELINE_COLOR,
        fontsize=8.6,
    )
    ax.set_xlabel("FT subset size (s)", fontweight="bold")
    ax.set_ylabel(r"$\mathrm{Var}(Y-\hat{Y})$ on raw rating scale", fontweight="bold")
    ax.set_title("")
    apply_reference_scaling_axis(ax)
    ax.set_xlim(x_min - 12, x_max + 12)
    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.91),
        handlelength=2.0,
        borderaxespad=0.2,
        fontsize=7.8,
    )
    fig.tight_layout(pad=0.25)
    preview_note(ax, has_replications)
    save_figure(fig, output_dir, f"method_comparison_{value_stem}_raw")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Wine loss scaling diagnostics.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=Path("Data/wine_data.csv"))
    parser.add_argument("--primary-method", default="var_stop_var")
    parser.add_argument("--value", choices=["auto", "validation_scale", "validation_stop"], default="validation_scale")
    parser.add_argument("--n-eff", type=float, default=8000.0)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "figures"
    plot_data = load_results(input_dir, args.data_path, args.value)
    summary = summarize(plot_data.frame, plot_data.has_replications)
    fits = fit_all(summary, plot_data.baseline_var, args.n_eff)

    primary_method = args.primary_method
    if primary_method not in set(summary["method"]):
        primary_method = str(summary.groupby("method")["mean"].mean().idxmin())

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"method_{plot_data.value_stem}_summary.csv", index=False)
    fits.to_csv(output_dir / "scaling_fit_parameters_raw.csv", index=False)

    setup_style()
    plot_linear_scaling(
        summary,
        fits,
        output_dir,
        primary_method,
        plot_data.has_replications,
        plot_data.value_label,
        plot_data.value_stem,
    )
    plot_loglog_scaling(
        summary,
        fits,
        output_dir,
        primary_method,
        plot_data.has_replications,
        plot_data.value_label,
        plot_data.value_stem,
    )
    plot_method_comparison(
        summary,
        output_dir,
        plot_data.baseline_var,
        plot_data.has_replications,
        plot_data.value_label,
        plot_data.value_stem,
    )

    print(f"input_dir={input_dir}")
    print(f"output_dir={output_dir}")
    print(f"value={plot_data.value_label}")
    print(f"has_replications={plot_data.has_replications}")
    print(f"primary_method={primary_method}")
    print("scaling fit parameters")
    print(fits.to_string(index=False))


if __name__ == "__main__":
    main()
