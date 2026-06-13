from __future__ import annotations

import numpy as np


def residual_variance_loss_numpy(y_true, pred) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(pred, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("y_true and pred must have the same length")
    residual = y - p
    return float(np.mean((residual - residual.mean()) ** 2))


def anchored_residual_variance_loss_numpy(y_true, pred, mean_penalty_weight: float = 0.0) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(pred, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("y_true and pred must have the same length")
    if mean_penalty_weight < 0:
        raise ValueError("mean_penalty_weight must be non-negative")
    residual = y - p
    variance_loss = residual_variance_loss_numpy(y_true, pred)
    return float(variance_loss + mean_penalty_weight * residual.mean() ** 2)
