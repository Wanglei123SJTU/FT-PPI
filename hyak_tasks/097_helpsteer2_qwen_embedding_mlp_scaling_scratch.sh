#!/bin/bash
set -euo pipefail

echo "helpsteer2_qwen_embedding_mlp_scaling_scratch_task_start $(date)"

REPO_DIR="${HYAK_RUNNER_REPO_DIR:-$PWD}"
cd "$REPO_DIR"
echo "repo=$(pwd)"
echo "head=$(git rev-parse --short HEAD 2>/dev/null || true)"

PYTHON_BIN=".venv-hyak/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

echo "python=$($PYTHON_BIN --version)"

GPU_ARGS="$(HYAK_GPU_MIN_IDLE="${HYAK_GPU_MIN_IDLE:-1}" bash scripts/choose_hyak_gpu.sh)"
echo "chosen_gpu_args=$GPU_ARGS"

PAIR_CSV="Data/helpsteer2_preference_pairs.csv"
EMBED_ROOT="artifacts/helpsteer2_preference/embeddings"
SCALING_ROOT="artifacts/helpsteer2_preference/mlp_scaling"
mkdir -p "$EMBED_ROOT" "$SCALING_ROOT" logs

echo "prepare HelpSteer2 preference pairs"
$PYTHON_BIN -m src.data.prepare_helpsteer2_preference \
  --output-csv "$PAIR_CSV"

echo "run local OLS screen before embedding"
$PYTHON_BIN -m src.experiments.helpsteer2_preference_regression \
  --input-csv "$PAIR_CSV" \
  --output-dir "artifacts/helpsteer2_preference/regression_screen"

submit_embedding_job() {
  local model_name="$1"
  local safe_name="$2"
  local max_length="$3"
  local batch_size="$4"
  local job_name="hs2_embed_${safe_name}"
  local embed_npz="${EMBED_ROOT}/${safe_name}_pair_embeddings.npz"
  local slurm_log="logs/${job_name}_%j.out"

  if [[ -f "$embed_npz" ]]; then
    echo "embedding_exists=$embed_npz"
    return 0
  fi

  echo "submit_embedding model=$model_name output=$embed_npz"
  # shellcheck disable=SC2086
  job_id=$(sbatch --parsable $GPU_ARGS \
    --job-name="$job_name" \
    --time=08:00:00 \
    --mem=120G \
    --cpus-per-task=8 \
    --output="$slurm_log" \
    --wrap="cd '$REPO_DIR' && source .venv-hyak/bin/activate && python -m src.experiments.helpsteer2_embedding_extraction --input-csv '$PAIR_CSV' --output-npz '$embed_npz' --model-name '$model_name' --batch-size '$batch_size' --max-length '$max_length'")
  echo "embedding_job_id=$job_id"

  while squeue -j "$job_id" -h | grep -q .; do
    squeue -j "$job_id" || true
    sleep 60
  done
  sacct -j "$job_id" --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS -P || true
  echo "embedding_log_tail"
  tail -n 120 logs/${job_name}_${job_id}.out || true

  [[ -f "$embed_npz" ]]
}

EMBED_NPZ=""
if submit_embedding_job "Qwen/Qwen2.5-72B-Instruct" "qwen2p5_72b" 768 1; then
  EMBED_NPZ="${EMBED_ROOT}/qwen2p5_72b_pair_embeddings.npz"
elif submit_embedding_job "Qwen/Qwen2.5-32B-Instruct" "qwen2p5_32b" 768 1; then
  EMBED_NPZ="${EMBED_ROOT}/qwen2p5_32b_pair_embeddings.npz"
elif submit_embedding_job "Qwen/Qwen2.5-14B-Instruct" "qwen2p5_14b" 768 2; then
  EMBED_NPZ="${EMBED_ROOT}/qwen2p5_14b_pair_embeddings.npz"
else
  echo "all_embedding_jobs_failed"
  exit 1
fi

echo "selected_embedding_npz=$EMBED_NPZ"

MLP_JOB_NAME="hs2_mlp_scaling"
MLP_OUT="${SCALING_ROOT}/$(basename "$EMBED_NPZ" .npz)_delta_format"
MLP_LOG="logs/${MLP_JOB_NAME}_%j.out"

echo "submit_mlp_scaling output=$MLP_OUT"
# shellcheck disable=SC2086
mlp_job_id=$(sbatch --parsable $GPU_ARGS \
  --job-name="$MLP_JOB_NAME" \
  --time=04:00:00 \
  --mem=64G \
  --cpus-per-task=8 \
  --output="$MLP_LOG" \
  --wrap="cd '$REPO_DIR' && source .venv-hyak/bin/activate && python -m src.experiments.helpsteer2_embedding_mlp_scaling --input-csv '$PAIR_CSV' --embedding-npz '$EMBED_NPZ' --output-dir '$MLP_OUT' --target delta_format --s-grid 0,50,100,250,500,750,1000,1500,3000 --replications 10 --hidden-dim 128 --batch-size 64 --max-epochs 300 --patience 30")
echo "mlp_job_id=$mlp_job_id"

while squeue -j "$mlp_job_id" -h | grep -q .; do
  squeue -j "$mlp_job_id" || true
  sleep 60
done
sacct -j "$mlp_job_id" --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS -P || true
echo "mlp_log_tail"
tail -n 160 logs/${MLP_JOB_NAME}_${mlp_job_id}.out || true

test -f "$MLP_OUT/mlp_scaling_summary.csv"
test -f "$MLP_OUT/budget_comparison.csv"
echo "mlp_scaling_summary"
cat "$MLP_OUT/mlp_scaling_summary.csv"
echo "budget_comparison"
cat "$MLP_OUT/budget_comparison.csv"
echo "mlp_scaling_report"
sed -n '1,220p' "$MLP_OUT/mlp_scaling_report.md"

echo "helpsteer2_qwen_embedding_mlp_scaling_scratch_task_done $(date)"
