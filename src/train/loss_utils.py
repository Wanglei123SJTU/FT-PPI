from __future__ import annotations

import numpy as np


def residual_variance_loss_numpy(y_true, pred) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(pred, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("y_true and pred must have the same length")
    residual = y - p
    return float(np.mean((residual - residual.mean()) ** 2))

