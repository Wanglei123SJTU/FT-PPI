#!/bin/bash
set -euo pipefail

echo "wine_full_grid_allocation_x8_task_start"
hostname
date
git rev-parse --short HEAD

CONFIG="${CONFIG:-configs/wine_full_grid_allocation.yaml}"
SLURM_TEMPLATE="${SLURM_TEMPLATE:-slurm/run_wine_full_grid_allocation.sbatch}"
SCRATCH_ROOT="/gscratch/scrubbed/$USER/ft-ppi"
SCRATCH_LOG_DIR="$SCRATCH_ROOT/logs"
mkdir -p "$SCRATCH_LOG_DIR" "$SCRATCH_ROOT/artifacts"

ROOT="$(
  .venv-hyak/bin/python - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(yaml.safe_load(f)["output_dir"])
PY
)"

echo "config=$CONFIG"
echo "output_root=$ROOT"
echo "slurm_template=$SLURM_TEMPLATE"

if [ ! -f Data/wine_data.csv ] && [ -f Code/wine_data.csv ]; then
  mkdir -p Data
  ln -s ../Code/wine_data.csv Data/wine_data.csv
  echo "created Data/wine_data.csv symlink to Code/wine_data.csv"
fi

echo "== CANCEL EXISTING WINE-FULLGRID JOBS =="
existing_jobs=$(squeue -u "$USER" -n wine-fullgrid -h -o "%A" 2>/dev/null | sort -u || true)
if [ -n "$existing_jobs" ]; then
  echo "$existing_jobs" | xargs -r scancel
  echo "cancelled existing jobs: $existing_jobs"
  sleep 20
else
  echo "no existing wine-fullgrid jobs"
fi

echo "== CLEAN OUTPUT FOR FULL-GRID ALLOCATION =="
rm -rf "$ROOT"
mkdir -p "$ROOT"

echo "== GPU STATUS =="
sinfo -o "%20P %18G %8D %8t %10C %10m %N" | grep -Ei 'gpu|ckpt|h200|a100|a40|l40|rtx6k' || true
squeue -u "$USER" || true

echo "== SUBMIT WINE FULL-GRID ARRAY ON BEST IDLE GPU =="
GPU_ARGS=$(HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-8}" bash scripts/choose_hyak_gpu.sh)
echo "GPU_ARGS=$GPU_ARGS"
PINNED_SBATCH="$SCRATCH_LOG_DIR/wine-fullgrid-${HYAK_RUNNER_TASK_ID:-manual}-$(date +%Y%m%d_%H%M%S).sbatch"
{
  echo "#!/bin/bash"
  printf '%s\n' "$GPU_ARGS" | tr ' ' '\n' | awk -F= '$1 == "--partition" {print "#SBATCH --partition="$2} $1 == "--gres" {print "#SBATCH --gres="$2}'
  tail -n +2 "$SLURM_TEMPLATE"
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
  echo "--- recent full-grid logs ---"
  find "$SCRATCH_LOG_DIR" -maxdepth 1 -name "wine-fullgrid-${JOB_ID}_*.out" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -8 | awk '{print $2}' | while read -r log; do
    [ -n "$log" ] || continue
    echo "--- tail $log ---"
    tail -80 "$log" || true
  done
  echo "--- partial cell count and time estimate ---"
  .venv-hyak/bin/python - "$CONFIG" <<'PY' || true
from pathlib import Path
import json
import statistics
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
root = Path(config["output_dir"])
done = list(root.glob("*/B_*/rep_*/rho_*/metrics.json"))
total = len(config["methods"]) * len(config["budgets"]) * len(config["replication_ids"]) * len(config["allocation_grid"])
runtimes = []
for path in done:
    try:
        with open(path, "r", encoding="utf-8") as f:
            runtimes.append(float(json.load(f)["runtime"]["runtime_seconds"]))
    except Exception:
        pass
print(f"completed_cells={len(done)} total_cells={total}")
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
find "$SCRATCH_LOG_DIR" -maxdepth 1 -name "wine-fullgrid-${JOB_ID}_*.out" -type f | sort | tail -16 | while read -r log; do
  echo "--- final tail $log ---"
  tail -100 "$log" || true
done

echo "== AGGREGATE AND VALIDATE =="
.venv-hyak/bin/python -m src.experiments.wine_full_grid_allocation aggregate --config "$CONFIG"

.venv-hyak/bin/python - "$CONFIG" <<'PY'
from pathlib import Path
import pandas as pd
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
root = Path(config["output_dir"])
required = [
    "full_grid_cell_metrics.csv",
    "full_grid_by_rho_summary.csv",
    "full_grid_allocation_summary.csv",
]
missing = [name for name in required if not (root / name).exists() or (root / name).stat().st_size == 0]
if missing:
    raise SystemExit("missing or empty required outputs: " + ", ".join(missing))
cell = pd.read_csv(root / "full_grid_cell_metrics.csv")
summary = pd.read_csv(root / "full_grid_allocation_summary.csv")
expected = len(config["methods"]) * len(config["budgets"]) * len(config["replication_ids"]) * len(config["allocation_grid"])
if len(cell) != expected:
    raise SystemExit(f"expected {expected} cells, got {len(cell)}")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
print("output_root", root)
print("cell_rows", len(cell))
print("runtime by method")
print(cell.groupby("method")[["runtime_seconds", "epochs_trained", "oom_fallback_used"]].agg({
    "runtime_seconds": "mean",
    "epochs_trained": "mean",
    "oom_fallback_used": "sum",
}).to_string())
print("allocation summary")
print(summary.to_string(index=False))
PY

echo "== SAVED RESULT FILES =="
find "$ROOT" -maxdepth 3 -type f \( -name "*.csv" -o -name "*.json" -o -name "*.pdf" -o -name "*.png" \) -printf "%s %p\n" | sort -n | tail -80

echo "wine_full_grid_allocation_x8_task_done"
