#!/bin/bash
set -euo pipefail

echo "upworthy_openai_embedding_mlp_pilot_task_start"
hostname
date
git rev-parse --short HEAD

SCRATCH_ROOT="/gscratch/scrubbed/$USER/ft-ppi"
SCRATCH_LOG_DIR="$SCRATCH_ROOT/logs"
mkdir -p "$SCRATCH_LOG_DIR"

CACHE_ZIP="$SCRATCH_ROOT/upworthy_openai_embedding_cache_upworthy_d512_n5000_20260621.zip"
FALLBACK_CACHE_ZIP="$PWD/upworthy_openai_embedding_cache_upworthy_d512_n5000_20260621.zip"
CACHE_DIR="$PWD/artifacts/upworthy_m_estimation/openai_embeddings"
CACHE_STEM="upworthy_openai_text-embedding-3-small_d512_n5000_seed20260621"
NPZ="$CACHE_DIR/${CACHE_STEM}_float16.npz"
SAMPLE_CSV="$CACHE_DIR/${CACHE_STEM}_sample.csv"
FEATURE_METADATA_CSV="$CACHE_DIR/${CACHE_STEM}_feature_metadata.csv"
SMOKE_OUT="$PWD/artifacts/upworthy_m_estimation/openai_embedding_mlp_smoke_${HYAK_RUNNER_TASK_ID:-149}"
FULL_OUT="$PWD/artifacts/upworthy_m_estimation/openai_embedding_mlp_pilot_${HYAK_RUNNER_TASK_ID:-149}"

echo "cache_zip=$CACHE_ZIP"
echo "fallback_cache_zip=$FALLBACK_CACHE_ZIP"
echo "cache_dir=$CACHE_DIR"
echo "smoke_out=$SMOKE_OUT"
echo "full_out=$FULL_OUT"

if [ ! -f "$CACHE_ZIP" ]; then
  if [ -f "$FALLBACK_CACHE_ZIP" ]; then
    CACHE_ZIP="$FALLBACK_CACHE_ZIP"
    echo "using fallback cache zip: $CACHE_ZIP"
  else
    echo "Missing cache zip: $CACHE_ZIP"
    echo "Also checked fallback: $FALLBACK_CACHE_ZIP"
    echo "Upload artifacts/upworthy_m_estimation/openai_embedding_cache_upworthy_d512_n5000_20260621.zip to $SCRATCH_ROOT/ before rerunning this task."
    exit 2
  fi
fi

mkdir -p "$CACHE_DIR"
.venv-hyak/bin/python - "$CACHE_ZIP" "$CACHE_DIR" "$NPZ" "$SAMPLE_CSV" "$FEATURE_METADATA_CSV" <<'PY'
from pathlib import Path
import sys
import zipfile

zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
required = [Path(arg) for arg in sys.argv[3:]]

if all(path.exists() and path.stat().st_size > 0 for path in required):
    print("embedding cache already unpacked")
    raise SystemExit(0)

out_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as zf:
    for member in zf.namelist():
        if member.endswith("/"):
            continue
        target = out_dir / Path(member).name
        with zf.open(member) as src, open(target, "wb") as dst:
            dst.write(src.read())
        print(f"unpacked {target} bytes={target.stat().st_size}")

missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
if missing:
    raise SystemExit("missing after unpack: " + ", ".join(missing))
PY

echo "== CACHE FILES =="
ls -lh "$NPZ" "$SAMPLE_CSV" "$FEATURE_METADATA_CSV"

echo "== GPU STATUS =="
sinfo -o "%20P %18G %8D %8t %10C %10m %N" | grep -Ei 'gpu|ckpt|h200|a100|a40|l40|rtx6k' || true
squeue -u "$USER" || true

echo "== SUBMIT OPENAI EMBEDDING MLP PILOT =="
if [ -n "${HYAK_FORCE_GPU_ARGS:-}" ]; then
  GPU_ARGS="$HYAK_FORCE_GPU_ARGS"
  echo "using forced GPU args"
else
  GPU_ARGS=$(HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-1}" bash scripts/choose_hyak_gpu.sh)
fi
echo "GPU_ARGS=$GPU_ARGS"

PINNED_SBATCH="$SCRATCH_LOG_DIR/upw-openai-mlp-${HYAK_RUNNER_TASK_ID:-149}-$(date +%Y%m%d_%H%M%S).sbatch"
{
  echo "#!/bin/bash"
  printf '%s\n' "$GPU_ARGS" | tr ' ' '\n' | awk -F= '$1 == "--partition" {print "#SBATCH --partition="$2} $1 == "--gres" {print "#SBATCH --gres="$2}'
  cat <<'SBATCH'
#SBATCH --job-name=upw-openai-mlp
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
SBATCH
  echo "#SBATCH --output=$SCRATCH_LOG_DIR/upw-openai-mlp-%j.out"
  cat <<SBATCH

set -euo pipefail
cd "$PWD"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "slurm_job_start \$(date)"
hostname
nvidia-smi || true
git rev-parse --short HEAD

module purge
module load cuda/12.4.1
module load foster/python/miniconda/3.8 2>/dev/null || \\
  module load chem/miniconda3/3.8 2>/dev/null || \\
  module load coenv/miniconda/3.13.11 2>/dev/null || \\
  module load coenv/python/3.11.9

VENV_DIR="\${HYAK_VENV_DIR:-.venv-hyak}"
VENV_REAL="\$(readlink -f "\$VENV_DIR" 2>/dev/null || echo "\$VENV_DIR")"
if [ ! -x "\$VENV_DIR/bin/python" ]; then
  echo "Missing \$VENV_DIR. Run: bash scripts/setup_hyak_env.sh"
  exit 1
fi
if [ -d "\$VENV_DIR/conda-meta" ] && command -v conda >/dev/null 2>&1; then
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "\$VENV_REAL"
else
  source "\$VENV_DIR/bin/activate"
fi
PYTHON_BIN="\$VENV_REAL/bin/python"
echo "PYTHON_BIN=\$PYTHON_BIN"

NPZ="$NPZ"
SAMPLE_CSV="$SAMPLE_CSV"
FEATURE_METADATA_CSV="$FEATURE_METADATA_CSV"
SMOKE_OUT="$SMOKE_OUT"
FULL_OUT="$FULL_OUT"

"\$PYTHON_BIN" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

echo "== SMOKE: delta_curiosity_share =="
"\$PYTHON_BIN" -m src.experiments.upworthy_embedding_mlp_screening \\
  --embedding-npz "\$NPZ" \\
  --sample-csv "\$SAMPLE_CSV" \\
  --feature-metadata-csv "\$FEATURE_METADATA_CSV" \\
  --output-dir "\$SMOKE_OUT" \\
  --targets delta_curiosity_share \\
  --s-grid 200,500,1000,3000 \\
  --budgets 500,1000,1500,3000,5000 \\
  --methods mse_stop_mse mse_stop_ifvar ifvar_stop_ifvar \\
  --replications 3 \\
  --hidden-dim 128 \\
  --batch-size 4096 \\
  --max-epochs 160 \\
  --patience 20 \\
  --device cuda

echo "== FULL PILOT: top targets =="
"\$PYTHON_BIN" -m src.experiments.upworthy_embedding_mlp_screening \\
  --embedding-npz "\$NPZ" \\
  --sample-csv "\$SAMPLE_CSV" \\
  --feature-metadata-csv "\$FEATURE_METADATA_CSV" \\
  --output-dir "\$FULL_OUT" \\
  --targets delta_curiosity_share,delta_vader_compound,delta_log_word_count \\
  --s-grid 200,500,1000,3000 \\
  --budgets 500,1000,1500,3000,5000 \\
  --methods mse_stop_mse mse_stop_ifvar ifvar_stop_ifvar \\
  --replications 20 \\
  --hidden-dim 128 \\
  --batch-size 4096 \\
  --max-epochs 200 \\
  --patience 25 \\
  --device cuda

echo "== PILOT SUMMARY =="
"\$PYTHON_BIN" - "\$FULL_OUT" <<'PY'
from pathlib import Path
import pandas as pd
import sys

root = Path(sys.argv[1])
required = [
    root / "embedding_mlp_summary.csv",
    root / "budget_win_summary.csv",
    root / "target_diagnostics.csv",
    root / "screening_report.md",
]
missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
if missing:
    raise SystemExit("missing required outputs: " + ", ".join(missing))

summary = pd.read_csv(root / "embedding_mlp_summary.csv")
budget = pd.read_csv(root / "budget_win_summary.csv")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
print("best variance reductions")
print(
    summary.sort_values(["target", "mean_ifvar_ratio_vs_zero"])
    .groupby("target", as_index=False)
    .head(5)
    .to_string(index=False)
)
print("\\nbest budget rows")
print(
    budget.sort_values(["target", "budget", "mean_ratio_vs_direct_ols"])
    .groupby(["target", "budget"], as_index=False)
    .head(1)
    .to_string(index=False)
)
PY

echo "slurm_job_done \$(date)"
SBATCH
} > "$PINNED_SBATCH"
chmod +x "$PINNED_SBATCH"
echo "PINNED_SBATCH=$PINNED_SBATCH"

submit_output=$(sbatch "$PINNED_SBATCH")
echo "$submit_output"
JOB_ID=$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/ {print $4}' | tail -1)
if [ -z "$JOB_ID" ]; then
  echo "Could not parse job id from sbatch output."
  exit 1
fi
echo "JOB_ID=$JOB_ID"

echo "== MONITOR JOB =="
while squeue -j "$JOB_ID" -h >/dev/null 2>&1 && [ -n "$(squeue -j "$JOB_ID" -h)" ]; do
  date
  squeue -j "$JOB_ID" || true
  log_file="$SCRATCH_LOG_DIR/upw-openai-mlp-${JOB_ID}.out"
  if [ -f "$log_file" ]; then
    echo "--- tail $log_file ---"
    tail -120 "$log_file" || true
  fi
  sleep 60
done

echo "== FINAL STATUS =="
sacct -j "$JOB_ID" --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,MaxRSS -P 2>/dev/null || true

LOG_FILE="$SCRATCH_LOG_DIR/upw-openai-mlp-${JOB_ID}.out"
if [ -f "$LOG_FILE" ]; then
  echo "== FINAL LOG TAIL =="
  tail -220 "$LOG_FILE" || true
fi

echo "== OUTPUT CHECK =="
.venv-hyak/bin/python - "$FULL_OUT" <<'PY'
from pathlib import Path
import pandas as pd
import sys

root = Path(sys.argv[1])
required = ["embedding_mlp_summary.csv", "budget_win_summary.csv", "target_diagnostics.csv", "screening_report.md"]
missing = [name for name in required if not (root / name).exists() or (root / name).stat().st_size == 0]
if missing:
    raise SystemExit("missing or empty required outputs: " + ", ".join(missing))

budget = pd.read_csv(root / "budget_win_summary.csv")
best = budget.sort_values(["target", "budget", "mean_ratio_vs_direct_ols"]).groupby(["target", "budget"], as_index=False).head(1)
print("full_out", root)
print(best.to_string(index=False))
PY

echo "upworthy_openai_embedding_mlp_pilot_task_done"
