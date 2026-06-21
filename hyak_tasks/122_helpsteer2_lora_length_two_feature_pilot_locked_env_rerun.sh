#!/bin/bash
set -euo pipefail

echo "helpsteer2_lora_length_two_feature_locked_env_rerun_start $(date)"

cd "${HYAK_RUNNER_REPO_DIR:-$PWD}"
bash scripts/setup_hyak_env.sh
bash hyak_tasks/101_helpsteer2_lora_length_two_feature_pilot.sh

echo "helpsteer2_lora_length_two_feature_locked_env_rerun_done $(date)"
