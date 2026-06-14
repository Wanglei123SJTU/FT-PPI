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
MIN_IDLE="${HYAK_GPU_MIN_IDLE:-8}"
GPU_CANDIDATES=()
GPU_CANDIDATE_KEYS=""

idle_gpu_count() {
  local partition="$1"
  local gpu_type="$2"
  sinfo -h -p "$partition" -o "%G|%D|%t" 2>/dev/null | awk -F'|' -v gpu="gpu:${gpu_type}:" '
    index($1, gpu) > 0 && $3 ~ /^idle/ {
      per_node = 0
      n = split($1, parts, ":")
      if (n >= 3) {
        per_node = parts[3] + 0
      }
      total += ($2 + 0) * per_node
    }
    END { print total + 0 }
  '
}

add_candidate() {
  local partition="$1"
  local gres="$2"
  local reason="$3"
  local key="${partition} ${gres}"
  case " $GPU_CANDIDATE_KEYS " in
    *" $key "*) return 0 ;;
  esac
  GPU_CANDIDATE_KEYS="$GPU_CANDIDATE_KEYS $key"
  GPU_CANDIDATES+=("${partition}|${gres}|${reason}")
}

add_candidate_from_args() {
  local args="$1"
  local reason="$2"
  local partition gres
  partition=$(printf '%s\n' "$args" | tr ' ' '\n' | awk -F= '$1 == "--partition" {print $2}' | tail -1)
  gres=$(printf '%s\n' "$args" | tr ' ' '\n' | awk -F= '$1 == "--gres" {print $2}' | tail -1)
  if [ -n "$partition" ] && [ -n "$gres" ]; then
    add_candidate "$partition" "$gres" "$reason"
  fi
}

add_candidate_if_idle() {
  local partition="$1"
  local gres="$2"
  local gpu_type="$3"
  local count
  count="$(idle_gpu_count "$partition" "$gpu_type")"
  if [ "$count" -ge "$MIN_IDLE" ]; then
    add_candidate "$partition" "$gres" "idle_${gpu_type}_${partition}_${count}"
  fi
}

gpu_args=$(HYAK_GPU_MIN_IDLE="$MIN_IDLE" bash scripts/choose_hyak_gpu.sh)
echo "GPU_ARGS=$gpu_args"
add_candidate_from_args "$gpu_args" "choose_hyak_gpu"
add_candidate_if_idle "gpu-h200" "gpu:h200:1" "h200"
add_candidate_if_idle "ckpt-g2" "gpu:h200:1" "h200"
add_candidate_if_idle "ckpt-all" "gpu:h200:1" "h200"
add_candidate_if_idle "gpu-a100" "gpu:a100:1" "a100"
add_candidate_if_idle "ckpt" "gpu:a100:1" "a100"
add_candidate_if_idle "ckpt-all" "gpu:a100:1" "a100"
add_candidate_if_idle "gpu-l40s" "gpu:l40s:1" "l40s"
add_candidate_if_idle "ckpt-g2" "gpu:l40s:1" "l40s"
add_candidate_if_idle "ckpt-all" "gpu:l40s:1" "l40s"
add_candidate_if_idle "gpu-l40" "gpu:l40:1" "l40"
add_candidate_if_idle "ckpt-g2" "gpu:l40:1" "l40"
add_candidate_if_idle "ckpt-all" "gpu:l40:1" "l40"
add_candidate_if_idle "gpu-a40" "gpu:a40:1" "a40"
add_candidate_if_idle "ckpt" "gpu:a40:1" "a40"
add_candidate_if_idle "ckpt-all" "gpu:a40:1" "a40"
add_candidate "ckpt" "gpu:a40:1" "last_resort"

echo "GPU_CANDIDATES:"
printf '  %s\n' "${GPU_CANDIDATES[@]}"

submit_output=""
submit_status=1
gpu_partition=""
gpu_gres=""
gpu_reason=""
for candidate in "${GPU_CANDIDATES[@]}"; do
  IFS='|' read -r gpu_partition gpu_gres gpu_reason <<< "$candidate"
  gpu_gres_safe="${gpu_gres//:/_}"
  PINNED_SBATCH="$SCRATCH_ROOT/logs/wine-lossctrl-${HYAK_RUNNER_TASK_ID:-manual}-${gpu_partition}-${gpu_gres_safe}-$(date +%Y%m%d_%H%M%S).sbatch"
  {
    echo "#!/bin/bash"
    echo "#SBATCH --partition=$gpu_partition"
    echo "#SBATCH --gres=$gpu_gres"
    tail -n +2 slurm/run_wine_loss_controlled_b10000.sbatch
  } > "$PINNED_SBATCH"
  chmod +x "$PINNED_SBATCH"
  echo "PINNED_SBATCH=$PINNED_SBATCH"
  echo "Trying candidate: partition=$gpu_partition gres=$gpu_gres reason=$gpu_reason"
  set +e
  submit_output=$(sbatch "$PINNED_SBATCH" 2>&1)
  submit_status=$?
  set -e
  echo "$submit_output"
  if [ "$submit_status" -eq 0 ]; then
    break
  fi
  echo "candidate_submit_failed partition=$gpu_partition gres=$gpu_gres reason=$gpu_reason"
done

if [ "$submit_status" -ne 0 ]; then
  echo "Wine loss controlled submit failed for all GPU candidates."
  exit 1
fi

JOB_ID=$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/ {print $4}' | tail -1)
if [ -z "$JOB_ID" ]; then
  echo "Could not parse job id from sbatch output."
  exit 1
fi
echo "JOB_ID=$JOB_ID"
echo "JOB_PROFILE=--partition=$gpu_partition --gres=$gpu_gres reason=$gpu_reason"

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
