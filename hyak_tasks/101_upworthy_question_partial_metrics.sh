#!/bin/bash
set -euo pipefail

TASK_ID="101_upworthy_question_partial_metrics"
LOCK_DIR=".hyak_runner/task_locks"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/${TASK_ID}.lock"
if ! flock -n 9; then
  echo "${TASK_ID}_already_running_elsewhere"
  exit 0
fi

echo "${TASK_ID}_task_start"
hostname
date
git rev-parse --short HEAD

.venv-hyak/bin/python - <<'PY'
from pathlib import Path
import json
import pandas as pd

roots = [
    Path("/gscratch/scrubbed/lei0603/ft-ppi/artifacts/upworthy_question_full5_current_balanced_focused_1p5b"),
    Path("/gscratch/scrubbed/lei0603/ft-ppi/artifacts/upworthy_question_vader_current_balanced_focused_1p5b"),
]
budgets = [300, 500, 1000, 1500, 2000, 3000, 5000]

for root in roots:
    print("\n== PARTIAL ROOT ==", root)
    paths = sorted(root.glob("*/rep_*/s_*/metrics.json"))
    print("metrics_count", len(paths))
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        scale = m.get("validation_scale", {})
        const = m.get("constant_validation_scale", {})
        inf = m.get("inference", {})
        runtime = m.get("runtime", {})
        direct = float(inf.get("direct_ols_ifvar_target_raw", float("nan")))
        ifvar = float(scale.get("if_residual_var_raw", float("nan")))
        const_ifvar = float(const.get("if_residual_var_raw", float("nan")))
        rows.append({
            "method": m.get("method"),
            "rep": int(m.get("replication_id")),
            "s": int(m.get("s_train")),
            "ifvar_raw": ifvar,
            "const_ifvar_raw": const_ifvar,
            "direct_ifvar_raw": direct,
            "ratio_to_const": ifvar / const_ifvar if const_ifvar else float("nan"),
            "ratio_to_direct": ifvar / direct if direct else float("nan"),
            "rmse_scaled": float(scale.get("rmse_scaled", float("nan"))),
            "corr": float(scale.get("correlation", float("nan"))),
            "epochs": int(m.get("epochs_trained", -1)),
            "runtime_sec": float(runtime.get("runtime_seconds", float("nan"))),
            "target_beta_raw": float(inf.get("target_coefficient_raw", inf.get("question_beta_raw", float("nan")))),
        })
    if not rows:
        print("no metrics yet")
        continue
    df = pd.DataFrame(rows).sort_values(["method", "rep", "s"])
    pd.set_option("display.max_rows", 120)
    pd.set_option("display.width", 240)
    print("\n-- completed cells --")
    print(df[["method", "rep", "s", "ifvar_raw", "ratio_to_const", "ratio_to_direct", "rmse_scaled", "corr", "epochs", "runtime_sec"]].to_string(index=False))

    by_s = (
        df.groupby(["method", "s"], as_index=False)
        .agg(
            n=("ifvar_raw", "size"),
            mean_ifvar=("ifvar_raw", "mean"),
            mean_ratio_const=("ratio_to_const", "mean"),
            mean_ratio_direct=("ratio_to_direct", "mean"),
            mean_rmse=("rmse_scaled", "mean"),
            mean_corr=("corr", "mean"),
            beta=("target_beta_raw", "first"),
            direct=("direct_ifvar_raw", "first"),
        )
        .sort_values(["method", "s"])
    )
    print("\n-- by method and s --")
    print(by_s.to_string(index=False))

    wins = []
    for _, row in by_s.iterrows():
        s = int(row["s"])
        if s <= 0:
            continue
        for B in budgets:
            if s >= B:
                continue
            direct_proxy = float(row["direct"]) / B
            ppi_proxy = float(row["mean_ifvar"]) / (B - s)
            wins.append({
                "B": B,
                "method": row["method"],
                "s": s,
                "n": int(row["n"]),
                "ppi_over_direct": ppi_proxy / direct_proxy if direct_proxy else float("nan"),
                "beats": ppi_proxy < direct_proxy,
            })
    win_df = pd.DataFrame(wins)
    print("\n-- temporary break-even proxy from partial cells --")
    if len(win_df):
        print(win_df.sort_values(["B", "ppi_over_direct"]).head(80).to_string(index=False))
        print("\n-- temporary winners --")
        tmp = win_df[win_df["beats"]].sort_values(["B", "ppi_over_direct"])
        print(tmp.to_string(index=False) if len(tmp) else "none")
    else:
        print("none")

print("\nactive queue snapshot")
PY

squeue -u "$USER" || true

echo "${TASK_ID}_task_done"
