#!/bin/bash
set -euo pipefail

echo "helpsteer2_lora_large_s_diagnostic_scratch_cache_task_start $(date)"

cd "${HYAK_RUNNER_REPO_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

CACHE_ROOT="/gscratch/scrubbed/${USER}/ft-ppi/cache"
mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/hf_datasets" "$CACHE_ROOT/torch" "$CACHE_ROOT/xdg"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf_datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

echo "cache_root=$CACHE_ROOT"
echo "hf_home=$HF_HOME"
echo "torch_home=$TORCH_HOME"

bash hyak_tasks/143_helpsteer2_lora_large_s_diagnostic.sh

echo "helpsteer2_lora_large_s_diagnostic_scratch_cache_task_done $(date)"
