#!/bin/bash
set -euo pipefail

echo "cancel_and_rerun_helpsteer2_lora_prewarm_start $(date)"

OLD_JOB_ID="36264554"
echo "cancelling_old_job=$OLD_JOB_ID"
scancel "$OLD_JOB_ID" || true

cd "${HYAK_RUNNER_REPO_DIR:-$PWD}"
export OUTPUT_DIR="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/helpsteer2_lora_length_two_feature_pilot_prewarm"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
bash hyak_tasks/101_helpsteer2_lora_length_two_feature_pilot.sh

echo "cancel_and_rerun_helpsteer2_lora_prewarm_done $(date)"
