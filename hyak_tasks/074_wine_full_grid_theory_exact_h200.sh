#!/bin/bash
set -euo pipefail

echo "wine_full_grid_theory_exact_task_start"
hostname
date
git rev-parse --short HEAD

CONFIG="${CONFIG:-configs/wine_full_grid_allocation_with_theory.yaml}"
SLURM_TEMPLATE="${SLURM_TEMPLATE:-slurm/run_wine_full_grid_theory_exact.sbatch}"
SCRATCH_ROOT="/gscratch/scrubbed/$USER/ft-ppi"
SCRATCH_LOG_DIR="$SCRATCH_ROOT/logs"
mkdir -p "$SCRATCH_LOG_DIR"

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
echo "This task only adds exact-theory LoRA-Var cells; it does not delete existing full-grid outputs."

if [ ! -f Data/wine_data.csv ] && [ -f Code/wine_data.csv ]; then
  mkdir -p Data
  ln -s ../Code/wine_data.csv Data/wine_data.csv
  echo "created Data/wine_data.csv symlink to Code/wine_data.csv"
fi

echo "== THEORY EXACT CELL PLAN =="
.venv-hyak/bin/python - "$CONFIG" <<'PY'
from src.experiments.wine_full_grid_allocation import load_config, theory_exact_task_cells, theoretical_allocation, effective_budget
import sys

config = load_config(sys.argv[1])
cells = theory_exact_task_cells(config)
print(f"theory_exact_cells={len(cells)}")
for budget in config["budgets"]:
    theory = theoretical_allocation(config, int(budget))
    print(
        "theory",
        f"B={budget}",
        f"n_eff={effective_budget(config, int(budget))}",
        f"s={theory['theory_s']}",
        f"rho={theory['theory_rho']:.9f}",
    )
PY

echo "== GPU STATUS =="
sinfo -o "%20P %18G %8D %8t %10C %10m %N" | grep -Ei 'gpu|ckpt|h200|a100|a40|l40|rtx6k' || true
squeue -u "$USER" || true

echo "== SUBMIT THEORY EXACT ARRAY =="
if [ -n "${HYAK_FORCE_GPU_ARGS:-}" ]; then
  GPU_ARGS="$HYAK_FORCE_GPU_ARGS"
  echo "using forced GPU args"
else
  GPU_ARGS=$(HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-8}" bash scripts/choose_hyak_gpu.sh)
fi
echo "GPU_ARGS=$GPU_ARGS"
PINNED_SBATCH="$SCRATCH_LOG_DIR/wine-theory-${HYAK_RUNNER_TASK_ID:-manual}-$(date +%Y%m%d_%H%M%S).sbatch"
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

echo "== MONITOR THEORY EXACT ARRAY =="
while squeue -j "$JOB_ID" -h >/dev/null 2>&1 && [ -n "$(squeue -j "$JOB_ID" -h)" ]; do
  date
  squeue -j "$JOB_ID" || true
  echo "--- recent theory logs ---"
  find "$SCRATCH_LOG_DIR" -maxdepth 1 -name "wine-theory-${JOB_ID}_*.out" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -8 | awk '{print $2}' | while read -r log; do
    [ -n "$log" ] || continue
    echo "--- tail $log ---"
    tail -80 "$log" || true
  done
  echo "--- exact theory cell count ---"
  .venv-hyak/bin/python - "$CONFIG" <<'PY' || true
from pathlib import Path
import sys
from src.experiments.wine_full_grid_allocation import load_config, theory_exact_task_cells, allocation_cell_dir

config = load_config(sys.argv[1])
done = 0
for method, budget, rep, rho in theory_exact_task_cells(config):
    if (allocation_cell_dir(config, method, budget, rep, rho) / "metrics.json").exists():
        done += 1
total = len(theory_exact_task_cells(config))
print(f"completed_theory_exact_cells={done} total_theory_exact_cells={total}")
PY
  sleep 180
done

echo "== FINAL STATUS =="
sacct -j "$JOB_ID" --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,MaxRSS -P 2>/dev/null || true

echo "== AGGREGATE WITH EXACT THEORY CELLS =="
.venv-hyak/bin/python -m src.experiments.wine_full_grid_allocation aggregate --config "$CONFIG"

.venv-hyak/bin/python - "$CONFIG" <<'PY'
from pathlib import Path
import pandas as pd
import sys
from src.experiments.wine_full_grid_allocation import load_config, theory_exact_task_cells, allocation_cell_dir

config = load_config(sys.argv[1])
root = Path(config["output_dir"])
missing = []
for method, budget, rep, rho in theory_exact_task_cells(config):
    path = allocation_cell_dir(config, method, budget, rep, rho) / "metrics.json"
    if not path.exists() or path.stat().st_size == 0:
        missing.append(str(path))
if missing:
    raise SystemExit("missing exact theory outputs:\n" + "\n".join(missing[:20]))

cell = pd.read_csv(root / "full_grid_cell_metrics.csv")
summary = pd.read_csv(root / "full_grid_allocation_summary.csv")
expected = 960 + len(theory_exact_task_cells(config))
if len(cell) != expected:
    raise SystemExit(f"expected {expected} aggregate rows with exact theory cells, got {len(cell)}")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
print("output_root", root)
print("cell_rows", len(cell))
print("exact_theory_rows", int((cell["allocation_source"] == "theory_exact").sum()))
print("allocation summary")
print(summary.to_string(index=False))
PY

echo "== SAVED EXACT THEORY RESULT FILES =="
find "$ROOT" -maxdepth 3 -type f \( -name "full_grid_*.csv" -o -path "*/figures/*.pdf" -o -path "*/figures/*.png" \) -printf "%s %p\n" | sort -n | tail -80

echo "wine_full_grid_theory_exact_task_done"
