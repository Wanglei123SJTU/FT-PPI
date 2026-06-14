#!/bin/bash
set -euo pipefail

echo "wine_loss_controlled_gpu_fallback_task_start"
hostname
date
git rev-parse --short HEAD

CONFIG="configs/wine_loss_controlled_b10000.yaml"
SCRATCH_ROOT="/gscratch/scrubbed/$USER/ft-ppi"
SCRATCH_LOG_DIR="$SCRATCH_ROOT/logs"
SCRATCH_ARTIFACT_DIR="$SCRATCH_ROOT/artifacts/wine_loss_controlled_b10000_r3_eval2000"
mkdir -p "$SCRATCH_LOG_DIR" "$SCRATCH_ROOT/artifacts"

echo "== CANCEL EXISTING WINE-LOSSCTRL JOBS =="
existing_jobs=$(squeue -u "$USER" -n wine-lossctrl -h -o "%A" 2>/dev/null | sort -u || true)
if [ -n "$existing_jobs" ]; then
  echo "$existing_jobs" | xargs -r scancel
  echo "cancelled existing jobs: $existing_jobs"
  sleep 20
else
  echo "no existing wine-lossctrl jobs"
fi

echo "== CLEAN OUTPUT FOR CONTROLLED COMPARISON =="
rm -rf "$SCRATCH_ARTIFACT_DIR"
echo "scratch_artifact_dir=$SCRATCH_ARTIFACT_DIR"
echo "scratch_log_dir=$SCRATCH_LOG_DIR"

echo "== GPU STATUS =="
sinfo -o "%20P %18G %8D %8t %10C %10m %N" | grep -Ei 'gpu|ckpt|h200|a100|a40|l40|rtx6k' || true
squeue -u "$USER" || true

echo "== SUBMIT WINE LOSS CONTROLLED ARRAY ON BEST IDLE GPU =="
gpu_args=$(HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-8}" bash scripts/choose_hyak_gpu.sh)
echo "GPU_ARGS=$gpu_args"
echo "Trying: sbatch $gpu_args slurm/run_wine_loss_controlled_b10000.sbatch"
set +e
submit_output=$(sbatch $gpu_args slurm/run_wine_loss_controlled_b10000.sbatch 2>&1)
submit_status=$?
set -e
echo "$submit_output"

if [ "$submit_status" -ne 0 ]; then
  echo "Best-idle GPU submit failed; falling back to ckpt gpu:1."
  gpu_args="--partition=ckpt --gres=gpu:1"
  set +e
  submit_output=$(sbatch $gpu_args slurm/run_wine_loss_controlled_b10000.sbatch 2>&1)
  submit_status=$?
  set -e
  echo "$submit_output"
  if [ "$submit_status" -ne 0 ]; then
    echo "Wine loss controlled fallback submit failed."
    exit 1
  fi
fi

JOB_ID=$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/ {print $4}' | tail -1)
if [ -z "$JOB_ID" ]; then
  echo "Could not parse job id from sbatch output."
  exit 1
fi
echo "JOB_ID=$JOB_ID"
echo "JOB_PROFILE=$gpu_args"

echo "== MONITOR ARRAY =="
while squeue -j "$JOB_ID" -h >/dev/null 2>&1 && [ -n "$(squeue -j "$JOB_ID" -h)" ]; do
  date
  squeue -j "$JOB_ID" || true
  echo "--- recent array logs ---"
  find "$SCRATCH_LOG_DIR" -maxdepth 1 -name "wine-lossctrl-${JOB_ID}_*.out" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -8 | awk '{print $2}' | while read -r log; do
    [ -n "$log" ] || continue
    echo "--- tail $log ---"
    tail -100 "$log" || true
  done
  sleep 180
done

echo "== FINAL STATUS =="
sacct -j "$JOB_ID" --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,MaxRSS -P 2>/dev/null || true

echo "== ARRAY LOG SUMMARY =="
find "$SCRATCH_LOG_DIR" -maxdepth 1 -name "wine-lossctrl-${JOB_ID}_*.out" -type f | sort | tail -16 | while read -r log; do
  echo "--- final tail $log ---"
  tail -100 "$log" || true
done

echo "== AGGREGATE AND VALIDATE =="
.venv-hyak/bin/python -m src.experiments.wine_var_scaling_law aggregate --config "$CONFIG"

.venv-hyak/bin/python - <<'PY'
from pathlib import Path
import json
import pandas as pd
import yaml

config_path = "configs/wine_loss_controlled_b10000.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
root = Path(config["output_dir"])
required = [
    "scaling_fit_by_rep.csv",
    "rampup_recovery_by_rep.csv",
    "rampup_recovery_summary.csv",
    "training_runtime_summary.csv",
    "loss_comparison_summary.csv",
    "leakage_check_report.json",
    "scaling_law_full_grid.pdf",
    "loss_comparison_residual_variance.pdf",
    "rampup_stagewise_fits.pdf",
    "rampup_regret_distribution.pdf",
]
missing = [name for name in required if not (root / name).exists() or (root / name).stat().st_size == 0]
if missing:
    raise SystemExit("missing or empty required outputs: " + ", ".join(missing))
with open(root / "leakage_check_report.json", "r", encoding="utf-8") as f:
    leak = json.load(f)
if not leak.get("passed"):
    raise SystemExit("leakage check did not pass")

runtime = pd.read_csv(root / "training_runtime_summary.csv")
expected_rows = len(config["losses"]) * len(config["replication_ids"]) * len(config["s_grid"])
if len(runtime) != expected_rows:
    raise SystemExit(f"expected {expected_rows} runtime rows, got {len(runtime)}")

fit = pd.read_csv(root / "scaling_fit_by_rep.csv")
ramp = pd.read_csv(root / "rampup_recovery_by_rep.csv")
ramp_summary = pd.read_csv(root / "rampup_recovery_summary.csv")
loss_summary = pd.read_csv(root / "loss_comparison_summary.csv")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)
print("output_root", root)
print("runtime rows", len(runtime))
print("runtime by loss")
print(runtime.groupby("loss")[["runtime_seconds", "epochs_trained", "early_stopped", "oom_fallback_used"]].agg({
    "runtime_seconds": "mean",
    "epochs_trained": "mean",
    "early_stopped": "mean",
    "oom_fallback_used": "sum",
}).to_string())
print("mean validation/eval residual variance by loss and s")
mean_cols = ["validation_scale_residual_var"]
if "eval_residual_var" in runtime.columns:
    mean_cols.append("eval_residual_var")
print(runtime.groupby(["loss", "s_train"], as_index=False)[mean_cols].mean().to_string(index=False))
print("loss comparison summary")
print(loss_summary.to_string(index=False))
print("scaling fit")
print(fit.to_string(index=False))
print("rampup by rep")
print(ramp.to_string(index=False))
print("rampup summary")
print(ramp_summary.to_string(index=False))
PY

echo "wine_loss_controlled_gpu_fallback_task_done"
