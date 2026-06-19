#!/bin/bash
set -euo pipefail

echo "upworthy_question_scaling_diagnostic_full_task_start"
hostname
date
git rev-parse --short HEAD

CONFIG="${CONFIG:-configs/upworthy_question_scaling_diagnostic_full.yaml}"
SLURM_TEMPLATE="${SLURM_TEMPLATE:-slurm/run_upworthy_question_scaling.sbatch}"
SCRATCH_ROOT="/gscratch/scrubbed/$USER/ft-ppi"
SCRATCH_LOG_DIR="$SCRATCH_ROOT/logs"
ARRAY_CONC="${ARRAY_CONC:-8}"
mkdir -p "$SCRATCH_LOG_DIR" "$SCRATCH_ROOT/artifacts"

ROOT="$(
  .venv-hyak/bin/python - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(yaml.safe_load(f)["output_dir"])
PY
)"

TOTAL_CELLS="$(
  .venv-hyak/bin/python - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
print(len(config["methods"]) * len(config["replication_ids"]) * len(config["s_grid"]))
PY
)"

if [ "$ARRAY_CONC" -gt "$TOTAL_CELLS" ]; then
  ARRAY_CONC="$TOTAL_CELLS"
fi

echo "config=$CONFIG"
echo "output_root=$ROOT"
echo "total_cells=$TOTAL_CELLS"
echo "array_concurrency=$ARRAY_CONC"
echo "slurm_template=$SLURM_TEMPLATE"

if [ ! -f Data/upworthy_pairs_with_text_features.csv ]; then
  echo "Missing Data/upworthy_pairs_with_text_features.csv"
  exit 1
fi

echo "== DESCRIBE DATA AND TARGET =="
.venv-hyak/bin/python -m src.experiments.upworthy_question_scaling_law describe --config "$CONFIG"

existing_cells="$(
  .venv-hyak/bin/python - "$CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
root = Path(config["output_dir"])
print(len(list(root.glob("*/rep_*/s_*/metrics.json"))))
PY
)"
if [ "$existing_cells" -ge "$TOTAL_CELLS" ]; then
  echo "== SKIP ARRAY: existing metrics complete =="
  echo "completed_cells=$existing_cells total_cells=$TOTAL_CELLS"
  goto_aggregate=1
else
  goto_aggregate=0
fi

echo "== GPU STATUS =="
sinfo -o "%20P %18G %8D %8t %10C %10m %N" | grep -Ei 'gpu|ckpt|h200|a100|a40|l40|rtx6k' || true
squeue -u "$USER" || true

if [ "$goto_aggregate" -eq 0 ]; then
echo "== SUBMIT UPWORTHY QUESTION SCALING DIAGNOSTIC =="
if [ -n "${HYAK_FORCE_GPU_ARGS:-}" ]; then
  GPU_ARGS="$HYAK_FORCE_GPU_ARGS"
  echo "using forced GPU args"
else
  GPU_ARGS=$(HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-$ARRAY_CONC}" bash scripts/choose_hyak_gpu.sh)
fi
echo "GPU_ARGS=$GPU_ARGS"

PINNED_SBATCH="$SCRATCH_LOG_DIR/upworthy-qscale-diagnostic-${HYAK_RUNNER_TASK_ID:-manual}-$(date +%Y%m%d_%H%M%S).sbatch"
{
  echo "#!/bin/bash"
  printf '%s\n' "$GPU_ARGS" | tr ' ' '\n' | awk -F= '$1 == "--partition" {print "#SBATCH --partition="$2} $1 == "--gres" {print "#SBATCH --gres="$2}'
  tail -n +2 "$SLURM_TEMPLATE" | sed "s/#SBATCH --array=0-7%4/#SBATCH --array=0-$((TOTAL_CELLS - 1))%${ARRAY_CONC}/"
} > "$PINNED_SBATCH"
chmod +x "$PINNED_SBATCH"
echo "PINNED_SBATCH=$PINNED_SBATCH"

export CONFIG
submit_output=$(sbatch "$PINNED_SBATCH")
echo "$submit_output"
JOB_ID=$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/ {print $4}' | tail -1)
if [ -z "$JOB_ID" ]; then
  echo "Could not parse job id from sbatch output."
  exit 1
fi
echo "JOB_ID=$JOB_ID"

echo "== MONITOR ARRAY =="
while squeue -j "$JOB_ID" -h >/dev/null 2>&1 && [ -n "$(squeue -j "$JOB_ID" -h)" ]; do
  date
  squeue -j "$JOB_ID" || true
  echo "--- recent upworthy diagnostic logs ---"
  find "$SCRATCH_LOG_DIR" -maxdepth 1 -name "upworthy-qscale-${JOB_ID}_*.out" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -10 | awk '{print $2}' | while read -r log; do
    [ -n "$log" ] || continue
    echo "--- tail $log ---"
    tail -80 "$log" || true
  done
  echo "--- partial cell count ---"
  .venv-hyak/bin/python - "$CONFIG" "$TOTAL_CELLS" <<'PY' || true
from pathlib import Path
import json
import statistics
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
total = int(sys.argv[2])
root = Path(config["output_dir"])
done = list(root.glob("*/rep_*/s_*/metrics.json"))
runtimes = []
by_method = {}
for path in done:
    try:
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        runtimes.append(float(metrics["runtime"]["runtime_seconds"]))
        by_method[metrics["method"]] = by_method.get(metrics["method"], 0) + 1
    except Exception:
        pass
print(f"completed_cells={len(done)} total_cells={total}")
print("completed_by_method=" + ", ".join(f"{k}:{v}" for k, v in sorted(by_method.items())) if by_method else "completed_by_method=none")
if runtimes:
    mean_sec = statistics.mean(runtimes)
    remaining = max(total - len(done), 0)
    print(f"mean_runtime_seconds={mean_sec:.1f} rough_remaining_gpu_hours={remaining * mean_sec / 3600:.2f}")
PY
  sleep 180
done

echo "== FINAL STATUS =="
sacct -j "$JOB_ID" --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,MaxRSS -P 2>/dev/null || true

echo "== ARRAY LOG SUMMARY =="
find "$SCRATCH_LOG_DIR" -maxdepth 1 -name "upworthy-qscale-${JOB_ID}_*.out" -type f | sort | tail -24 | while read -r log; do
  echo "--- final tail $log ---"
  tail -100 "$log" || true
done
fi

echo "== AGGREGATE AND VALIDATE =="
.venv-hyak/bin/python -m src.experiments.upworthy_question_scaling_law aggregate --config "$CONFIG"

.venv-hyak/bin/python - "$CONFIG" "$TOTAL_CELLS" <<'PY'
from pathlib import Path
import pandas as pd
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
total = int(sys.argv[2])
root = Path(config["output_dir"])
required = [
    "scaling_cell_metrics.csv",
    "scaling_by_s_summary.csv",
    "scaling_fit_parameters_raw.csv",
    "break_even_diagnostics.csv",
]
missing = [name for name in required if not (root / name).exists() or (root / name).stat().st_size == 0]
if missing:
    raise SystemExit("missing or empty required outputs: " + ", ".join(missing))
cell = pd.read_csv(root / "scaling_cell_metrics.csv")
by_s = pd.read_csv(root / "scaling_by_s_summary.csv")
fits = pd.read_csv(root / "scaling_fit_parameters_raw.csv")
break_even = pd.read_csv(root / "break_even_diagnostics.csv")
if len(cell) != total:
    raise SystemExit(f"expected {total} cells, got {len(cell)}")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)
print("output_root", root)
print("cell_rows", len(cell))
print("target beta and direct benchmark")
print(cell[["question_beta_raw", "direct_ols_ifvar_target_raw"]].drop_duplicates().to_string(index=False))
print("best by IF variance at each s")
cols = [
    "s_train",
    "method",
    "sampling_strategy",
    "train_backbone",
    "mean_ifvarq_raw",
    "constant_ifvarq_raw",
    "ratio_to_constant_ifvarq",
    "drop_from_constant_pct",
    "ratio_to_direct_ols_ifvarq",
    "mean_mse_scaled",
    "mean_rmse_scaled",
    "mean_corr",
    "mean_epochs_trained",
]
cols = [col for col in cols if col in by_s.columns]
print(by_s.sort_values(["s_train", "mean_ifvarq_raw"]).groupby("s_train").head(5)[cols].to_string(index=False))
print("scaling fit")
print(fits.sort_values("r2", ascending=False).to_string(index=False))
print("break-even winners")
winners = break_even.loc[break_even["beats_labeled_only_proxy"]].sort_values(["budget_B", "var_ratio_to_labeled_only"])
print(winners.head(60).to_string(index=False) if len(winners) else "no break-even winners")
PY

echo "== SAVED RESULT FILES =="
find "$ROOT" -maxdepth 3 -type f \( -name "*.csv" -o -name "*.json" \) -printf "%s %p\n" | sort -n | tail -100

echo "upworthy_question_scaling_diagnostic_full_task_done"
