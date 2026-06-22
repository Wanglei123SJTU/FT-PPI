#!/bin/bash
set -euo pipefail

OUT_DIR="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/helpsteer2_lora_length_two_feature_pilot_fixed_fast"

echo "summarize_helpsteer2_lora_partial_start $(date)"
echo "out_dir=$OUT_DIR"
echo "cell_count=$(find "$OUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | wc -l | tr -d ' ')"

python - <<'PY'
import json
from pathlib import Path

import pandas as pd

out_dir = Path("/gscratch/scrubbed/lei0603/ft-ppi/artifacts/helpsteer2_lora_length_two_feature_pilot_fixed_fast")
paths = sorted((out_dir / "cells").glob("cell_*.json"))
rows = [json.loads(path.read_text()) for path in paths]
if not rows:
    print("no completed cells yet")
    raise SystemExit(0)

df = pd.DataFrame(rows).sort_values(["replication", "s", "method"])
cols = [
    "task_index",
    "replication",
    "s",
    "method",
    "best_epoch",
    "epochs_run",
    "eval_ifvar",
    "baseline_eval_ifvar",
    "eval_mse",
    "seconds",
]
print(df[cols].to_string(index=False))
print()
summary = (
    df.groupby(["s", "method"], as_index=False)
    .agg(
        n=("eval_ifvar", "size"),
        eval_ifvar_mean=("eval_ifvar", "mean"),
        baseline=("baseline_eval_ifvar", "mean"),
        seconds_mean=("seconds", "mean"),
    )
    .sort_values(["s", "method"])
)
summary["ratio_to_baseline"] = summary["eval_ifvar_mean"] / summary["baseline"]
print(summary.to_string(index=False))
PY

echo "summarize_helpsteer2_lora_partial_done $(date)"
