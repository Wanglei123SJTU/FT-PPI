#!/bin/bash
set -euo pipefail

OLD_JOB_ID="36266039"
NEW_OUTPUT_DIR="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/helpsteer2_lora_length_two_feature_pilot_fixed_fast"

echo "cancel_and_rerun_helpsteer2_lora_fixed_fast_start $(date)"
echo "old_job_id=$OLD_JOB_ID"

if squeue -j "$OLD_JOB_ID" -h | grep -q .; then
  echo "canceling old job $OLD_JOB_ID"
  scancel "$OLD_JOB_ID" || true
else
  echo "old job $OLD_JOB_ID is not running"
fi

mkdir -p "$NEW_OUTPUT_DIR"

OUTPUT_DIR="$NEW_OUTPUT_DIR" \
MAX_LENGTH=512 \
TRAIN_BATCH_SIZE=8 \
EVAL_BATCH_SIZE=16 \
GRAD_ACCUM=2 \
MAX_EPOCHS=5 \
PATIENCE=1 \
NO_GRADIENT_CHECKPOINTING=1 \
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-8}" \
bash hyak_tasks/101_helpsteer2_lora_length_two_feature_pilot.sh

echo "cancel_and_rerun_helpsteer2_lora_fixed_fast_done $(date)"
