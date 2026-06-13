from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


EPS = 1e-12


@dataclass
class EstimateResult:
    method: str
    estimate: float
    estimated_se: float
    ci_low: float
    ci_high: float
    ci_length: float
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | str]:
        row: dict[str, float | str] = {
            "method": self.method,
            "estimate": self.estimate,
            "estimated_se": self.estimated_se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_length": self.ci_length,
        }
        row.update(self.extra)
        return row


def _as_1d(x: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty input")
    return arr


def _var(x: np.ndarray) -> float:
    return float(np.var(x, ddof=1)) if len(x) > 1 else 0.0


def _ci(estimate: float, variance: float, method: str, extra: dict[str, float] | None = None) -> EstimateResult:
    se = float(np.sqrt(max(variance, 0.0)))
    low = estimate - 1.96 * se
    high = estimate + 1.96 * se
    return EstimateResult(
        method=method,
        estimate=float(estimate),
        estimated_se=se,
        ci_low=float(low),
        ci_high=float(high),
        ci_length=float(high - low),
        extra=extra or {},
    )


def sample_mean_estimate(y_labeled: np.ndarray | list[float], method: str = "sample_mean") -> EstimateResult:
    y = _as_1d(y_labeled)
    estimate = float(np.mean(y))
    variance = _var(y) / len(y)
    return _ci(estimate, variance, method, {"n_labeled": float(len(y))})


def ppi_estimate(
    y_correction: np.ndarray | list[float],
    f_correction: np.ndarray | list[float],
    f_unlabeled: np.ndarray | list[float],
    lambda_value: float = 1.0,
    method: str = "ppi",
) -> EstimateResult:
    y_c = _as_1d(y_correction)
    f_c = _as_1d(f_correction)
    f_u = _as_1d(f_unlabeled)
    if len(y_c) != len(f_c):
        raise ValueError("y_correction and f_correction must have the same length")

    lam = float(lambda_value)
    residual = y_c - lam * f_c
    estimate = float(np.mean(y_c) + lam * (np.mean(f_u) - np.mean(f_c)))
    variance = _var(residual) / len(residual) + (lam**2) * _var(f_u) / len(f_u)
    return _ci(
        estimate,
        variance,
        method,
        {
            "lambda": lam,
            "n_correction": float(len(y_c)),
            "n_unlabeled": float(len(f_u)),
        },
    )


def estimate_ppi_plus_lambda(
    y_correction: np.ndarray | list[float],
    f_correction: np.ndarray | list[float],
    n_unlabeled: int,
    eps: float = EPS,
) -> float:
    """Finite-sample PPI++ control-variate coefficient.

    If the predictor has near-zero correction variance, fall back to vanilla PPI
    with lambda = 1.0. This keeps the estimator finite and conservative for
    smoke tests.
    """
    y_c = _as_1d(y_correction)
    f_c = _as_1d(f_correction)
    if len(y_c) != len(f_c):
        raise ValueError("y_correction and f_correction must have the same length")
    var_f = _var(f_c)
    if var_f <= eps:
        return 1.0
    cov = float(np.cov(y_c, f_c, ddof=1)[0, 1]) if len(y_c) > 1 else 0.0
    shrink = n_unlabeled / (n_unlabeled + len(y_c))
    return float((cov / var_f) * shrink)


def ppi_plus_estimate(
    y_correction: np.ndarray | list[float],
    f_correction: np.ndarray | list[float],
    f_unlabeled: np.ndarray | list[float],
    method: str = "ppi_plus",
) -> EstimateResult:
    f_u = _as_1d(f_unlabeled)
    lam = estimate_ppi_plus_lambda(y_correction, f_correction, len(f_u))
    return ppi_estimate(y_correction, f_correction, f_u, lambda_value=lam, method=method)


def residual_variance_value(y_true: np.ndarray | list[float], pred: np.ndarray | list[float]) -> float:
    y = _as_1d(y_true)
    p = _as_1d(pred)
    if len(y) != len(p):
        raise ValueError("y_true and pred must have the same length")
    residual = y - p
    centered = residual - residual.mean()
    return float(np.mean(centered**2))


def attach_truth_metrics(row: dict[str, float | str], true_mean: float) -> dict[str, float | str]:
    estimate = float(row["estimate"])
    se = float(row["estimated_se"])
    out = dict(row)
    out["bias"] = estimate - true_mean
    out["rmse"] = abs(estimate - true_mean)
    out["coverage"] = float(float(row["ci_low"]) <= true_mean <= float(row["ci_high"]))
    out["estimated_variance"] = se**2
    return out

