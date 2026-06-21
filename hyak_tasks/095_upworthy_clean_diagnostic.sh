#!/bin/bash
set -euo pipefail

echo "upworthy_clean_diagnostic_task_start $(date)"
hostname
git rev-parse --short HEAD

SCRATCH_ROOT="/gscratch/scrubbed/$USER/ft-ppi"
SCRATCH_LOG_DIR="$SCRATCH_ROOT/logs"
SLURM_TEMPLATE="${SLURM_TEMPLATE:-slurm/run_upworthy_question_scaling.sbatch}"
ARRAY_CONC="${ARRAY_CONC:-8}"
CONFIGS="${CONFIGS:-configs/upworthy_question_vader_clean_diagnostic_1p5b.yaml configs/upworthy_vader_question_clean_diagnostic_1p5b.yaml}"
mkdir -p "$SCRATCH_LOG_DIR" "$SCRATCH_ROOT/artifacts"

if [ ! -x .venv-hyak/bin/python ]; then
  echo "Missing .venv-hyak/bin/python"
  exit 1
fi

if [ ! -f Data/upworthy_pairs_with_text_features.csv ]; then
  echo "Missing Data/upworthy_pairs_with_text_features.csv"
  exit 1
fi

echo "== GPU STATUS BEFORE SUBMISSION =="
sinfo -o "%20P %18G %8D %8t %10C %10m %N" | grep -Ei 'gpu|ckpt|h200|a100|a40|l40|rtx6k' || true
squeue -u "$USER" || true

for CONFIG in $CONFIGS; do
  echo "== CLEAN DIAGNOSTIC CONFIG: $CONFIG =="
  if [ ! -f "$CONFIG" ]; then
    echo "Missing config: $CONFIG"
    exit 1
  fi

  ROOT="$(
    .venv-hyak/bin/python - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(yaml.safe_load(f)["output_dir"])
PY
  )"
  TOTAL_CELLS="$(
    .venv-hyak/bin/python - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
print(len(config["methods"]) * len(config["replication_ids"]) * len(config["s_grid"]))
PY
  )"
  THIS_ARRAY_CONC="$ARRAY_CONC"
  if [ "$THIS_ARRAY_CONC" -gt "$TOTAL_CELLS" ]; then
    THIS_ARRAY_CONC="$TOTAL_CELLS"
  fi
  echo "output_root=$ROOT"
  echo "total_cells=$TOTAL_CELLS"
  echo "array_concurrency=$THIS_ARRAY_CONC"

  echo "== DESCRIBE =="
  .venv-hyak/bin/python -m src.experiments.upworthy_question_scaling_law describe --config "$CONFIG"

  existing_cells="$(
    .venv-hyak/bin/python - "$CONFIG" <<'PY'
from pathlib import Path
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
root = Path(config["output_dir"])
print(len(list(root.glob("*/rep_*/s_*/metrics.json"))))
PY
  )"

  if [ "$existing_cells" -lt "$TOTAL_CELLS" ]; then
    echo "== SUBMIT ARRAY existing_cells=$existing_cells total_cells=$TOTAL_CELLS =="
    if [ -n "${HYAK_FORCE_GPU_ARGS:-}" ]; then
      GPU_ARGS="$HYAK_FORCE_GPU_ARGS"
      echo "using forced GPU args"
    else
      GPU_ARGS=$(HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-$THIS_ARRAY_CONC}" bash scripts/choose_hyak_gpu.sh)
    fi
    echo "GPU_ARGS=$GPU_ARGS"

    SAFE_NAME="$(basename "$CONFIG" .yaml | tr -cd 'A-Za-z0-9_-.')"
    PINNED_SBATCH="$SCRATCH_LOG_DIR/upworthy-clean-${SAFE_NAME}-${HYAK_RUNNER_TASK_ID:-manual}-$(date +%Y%m%d_%H%M%S).sbatch"
    {
      echo "#!/bin/bash"
      printf '%s\n' "$GPU_ARGS" | tr ' ' '\n' | awk -F= '$1 == "--partition" {print "#SBATCH --partition="$2} $1 == "--gres" {print "#SBATCH --gres="$2}'
      tail -n +2 "$SLURM_TEMPLATE" | sed "s/#SBATCH --array=0-7%4/#SBATCH --array=0-$((TOTAL_CELLS - 1))%${THIS_ARRAY_CONC}/"
    } > "$PINNED_SBATCH"
    chmod +x "$PINNED_SBATCH"
    echo "PINNED_SBATCH=$PINNED_SBATCH"

    export CONFIG
    submit_output=$(sbatch "$PINNED_SBATCH")
    echo "$submit_output"
    JOB_ID=$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/ {print $4}' | tail -1)
    if [ -z "$JOB_ID" ]; then
      echo "Could not parse job id"
      exit 1
    fi
    echo "JOB_ID=$JOB_ID"

    while squeue -j "$JOB_ID" -h >/dev/null 2>&1 && [ -n "$(squeue -j "$JOB_ID" -h)" ]; do
      date
      squeue -j "$JOB_ID" || true
      echo "--- recent clean diagnostic logs ---"
      find "$SCRATCH_LOG_DIR" -maxdepth 1 -name "upworthy-qscale-${JOB_ID}_*.out" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -8 | awk '{print $2}' | while read -r log; do
        [ -n "$log" ] || continue
        echo "--- tail $log ---"
        tail -60 "$log" || true
      done
      echo "--- partial cell count ---"
      .venv-hyak/bin/python - "$CONFIG" "$TOTAL_CELLS" <<'PY' || true
from pathlib import Path
import json, statistics, sys, yaml
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
    print(f"mean_runtime_seconds={mean_sec:.1f} rough_remaining_gpu_hours={max(total-len(done),0)*mean_sec/3600:.2f}")
PY
      current_cells="$(
        .venv-hyak/bin/python - "$CONFIG" <<'PY'
from pathlib import Path
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
print(len(list(Path(config["output_dir"]).glob("*/rep_*/s_*/metrics.json"))))
PY
      )"
      if [ "$current_cells" -ge "$TOTAL_CELLS" ]; then
        echo "metrics_complete_before_slurm_exit current_cells=$current_cells total_cells=$TOTAL_CELLS"
        scancel "$JOB_ID" 2>/dev/null || true
        break
      fi
      sleep 180
    done

    echo "== FINAL SLURM STATUS =="
    sacct -j "$JOB_ID" --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,MaxRSS -P 2>/dev/null || true
  else
    echo "== SKIP ARRAY: metrics already complete =="
  fi

  echo "== AGGREGATE $CONFIG =="
  .venv-hyak/bin/python -m src.experiments.upworthy_question_scaling_law aggregate --config "$CONFIG"
  .venv-hyak/bin/python - "$CONFIG" "$TOTAL_CELLS" <<'PY'
from pathlib import Path
import pandas as pd
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
total = int(sys.argv[2])
root = Path(config["output_dir"])
cell = pd.read_csv(root / "scaling_cell_metrics.csv")
by_s = pd.read_csv(root / "scaling_by_s_summary.csv")
fits = pd.read_csv(root / "scaling_fit_parameters_raw.csv")
break_even = pd.read_csv(root / "break_even_diagnostics.csv")
if len(cell) != total:
    raise SystemExit(f"expected {total} cells, got {len(cell)}")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)
print("output_root", root)
print("target summary")
print(cell[["target_feature", "target_coefficient_raw", "direct_ols_ifvar_target_raw", "validation_scale_size", "validation_scale_source", "effective_batch_size"]].drop_duplicates().to_string(index=False))
print("best methods by s")
cols = [
    "s_train",
    "method",
    "sampling_strategy",
    "mean_ifvarq_raw",
    "median_ifvarq_raw",
    "ratio_to_direct_ols_ifvarq",
    "median_ratio_to_direct_ols_ifvarq",
    "mean_mse_scaled",
    "mean_corr",
    "mean_epochs_trained",
]
print(by_s.sort_values(["s_train", "mean_ifvarq_raw"]).groupby("s_train").head(4)[cols].to_string(index=False))
print("scaling fit")
print(fits.sort_values("r2", ascending=False).to_string(index=False))
print("break-even winners")
winners = break_even.loc[break_even["beats_labeled_only_proxy"]].sort_values(["budget_B", "var_ratio_to_labeled_only"])
print(winners.head(80).to_string(index=False) if len(winners) else "no break-even winners")
PY
done

echo "upworthy_clean_diagnostic_task_done $(date)"
