#!/bin/bash
set -euo pipefail

echo "helpsteer2_lora_length_two_feature_pilot_rerun_task_start $(date)"
exec bash hyak_tasks/101_helpsteer2_lora_length_two_feature_pilot.sh
