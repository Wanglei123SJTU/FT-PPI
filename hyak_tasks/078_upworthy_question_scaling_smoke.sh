#!/bin/bash
set -euo pipefail

echo "upworthy_question_scaling_smoke_task_start"
hostname
date
git rev-parse --short HEAD

CONFIG="${CONFIG:-configs/upworthy_question_scaling_smoke.yaml}"
SLURM_TEMPLATE="${SLURM_TEMPLATE:-slurm/run_upworthy_question_scaling.sbatch}"
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

TOTAL_CELLS="$(
  .venv-hyak/bin/python - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
print(len(config["methods"]) * len(config["replication_ids"]) * len(config["s_grid"]))
PY
)"

echo "config=$CONFIG"
echo "output_root=$ROOT"
echo "total_cells=$TOTAL_CELLS"
echo "slurm_template=$SLURM_TEMPLATE"

if [ ! -f Data/upworthy_pairs_with_text_features.csv ]; then
  echo "Missing Data/upworthy_pairs_with_text_features.csv"
  exit 1
fi

echo "== DESCRIBE DATA AND TARGET =="
.venv-hyak/bin/python -m src.experiments.upworthy_question_scaling_law describe --config "$CONFIG"

echo "== GPU STATUS =="
sinfo -o "%20P %18G %8D %8t %10C %10m %N" | grep -Ei 'gpu|ckpt|h200|a100|a40|l40|rtx6k' || true
squeue -u "$USER" || true

echo "== SUBMIT UPWORTHY QUESTION SCALING SMOKE =="
if [ -n "${HYAK_FORCE_GPU_ARGS:-}" ]; then
  GPU_ARGS="$HYAK_FORCE_GPU_ARGS"
  echo "using forced GPU args"
else
  GPU_ARGS=$(HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-4}" bash scripts/choose_hyak_gpu.sh)
fi
echo "GPU_ARGS=$GPU_ARGS"

PINNED_SBATCH="$SCRATCH_LOG_DIR/upworthy-qscale-${HYAK_RUNNER_TASK_ID:-manual}-$(date +%Y%m%d_%H%M%S).sbatch"
{
  echo "#!/bin/bash"
  printf '%s\n' "$GPU_ARGS" | tr ' ' '\n' | awk -F= '$1 == "--partition" {print "#SBATCH --partition="$2} $1 == "--gres" {print "#SBATCH --gres="$2}'
  tail -n +2 "$SLURM_TEMPLATE" | sed "s/#SBATCH --array=0-7%4/#SBATCH --array=0-$((TOTAL_CELLS - 1))%4/"
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
  echo "--- recent upworthy scaling logs ---"
  find "$SCRATCH_LOG_DIR" -maxdepth 1 -name "upworthy-qscale-${JOB_ID}_*.out" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -8 | awk '{print $2}' | while read -r log; do
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
find "$SCRATCH_LOG_DIR" -maxdepth 1 -name "upworthy-qscale-${JOB_ID}_*.out" -type f | sort | tail -16 | while read -r log; do
  echo "--- final tail $log ---"
  tail -100 "$log" || true
done

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
pd.set_option("display.width", 240)
print("output_root", root)
print("cell_rows", len(cell))
print("target beta and direct benchmark")
print(cell[["question_beta_raw", "direct_ols_ifvar_target_raw"]].drop_duplicates().to_string(index=False))
print("by-s summary")
print(by_s[["method", "s_train", "mean_ifvarq_raw", "mean_mse_scaled", "mean_rmse_scaled", "mean_corr"]].to_string(index=False))
print("scaling fit")
print(fits.to_string(index=False))
print("break-even diagnostics")
print(break_even.to_string(index=False))
PY

echo "== SAVED RESULT FILES =="
find "$ROOT" -maxdepth 3 -type f \( -name "*.csv" -o -name "*.json" \) -printf "%s %p\n" | sort -n | tail -80

echo "upworthy_question_scaling_smoke_task_done"
