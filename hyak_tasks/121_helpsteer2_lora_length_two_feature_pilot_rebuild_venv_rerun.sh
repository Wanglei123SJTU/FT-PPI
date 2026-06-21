#!/bin/bash
set -euo pipefail

echo "helpsteer2_lora_length_two_feature_rebuild_venv_rerun_start $(date)"
RESET=1 FORCE_INSTALL=1 bash scripts/setup_hyak_env.sh
bash hyak_tasks/101_helpsteer2_lora_length_two_feature_pilot.sh
echo "helpsteer2_lora_length_two_feature_rebuild_venv_rerun_done $(date)"
