from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_prediction_frame(
    output_path: str | Path,
    population: pd.DataFrame,
    pred_scaled,
    label_mean: float,
    label_std: float,
    method: str,
    model_name: str,
    loss: str,
) -> pd.DataFrame:
    pred_scaled_arr = np.asarray(pred_scaled, dtype=float).reshape(-1)
    if len(pred_scaled_arr) != len(population):
        raise ValueError("prediction length must match population length")
    out = pd.DataFrame(
        {
            "sample_id": population["sample_id"].to_numpy(),
            "split_role": population["split_role"].to_numpy(),
            "y_true": population["points"].astype(float).to_numpy(),
            "pred_scaled": pred_scaled_arr,
            "pred_mean": pred_scaled_arr * label_std + label_mean,
            "method": method,
            "model_name": model_name,
            "loss": loss,
        }
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out

