#!/bin/bash
set -euo pipefail

echo "inspect_helpsteer2_worker_quickdiag_task_start $(date)"

cd "${HYAK_RUNNER_REPO_DIR:-$PWD}"
echo "repo=$(pwd)"
git status --short --branch
echo "commit=$(git rev-parse --short HEAD)"

STATE_DIR="${HYAK_RUNNER_STATE_DIR:-.hyak_runner}"
RUNNER_LOG_DIR="/gscratch/scrubbed/${USER}/ft-ppi/runner_logs"
OUTPUT_DIR="/gscratch/scrubbed/${USER}/ft-ppi/artifacts/helpsteer2_lora_worker_quickdiag"

echo "state_dir=$STATE_DIR"
echo "runner_log_dir=$RUNNER_LOG_DIR"
echo "output_dir=$OUTPUT_DIR"

echo "runner markers"
for kind in running done failed; do
  echo "-- $kind --"
  ls -la "$STATE_DIR/$kind" 2>/dev/null || true
  for marker in "$STATE_DIR/$kind"/132_helpsteer2_lora_worker_quickdiag; do
    if [ -e "$marker" ]; then
      echo "marker=$marker"
      cat "$marker" || true
    fi
  done
done

echo "runner/task processes"
ps -u "$USER" -o pid,ppid,stat,etime,cmd | grep -E 'hyak_runner|132_helpsteer2|run_helpsteer2_lora_worker|helpsteer2_lora_scaling|sbatch|squeue' | grep -v grep || true

echo "latest 132 runner task logs"
ls -1t "$RUNNER_LOG_DIR"/132_helpsteer2_lora_worker_quickdiag_*.log 2>/dev/null | head -n 5 || true
latest_132="$(ls -1t "$RUNNER_LOG_DIR"/132_helpsteer2_lora_worker_quickdiag_*.log 2>/dev/null | head -n 1 || true)"
if [ -n "$latest_132" ]; then
  echo "---- tail latest 132 log: $latest_132 ----"
  tail -n 240 "$latest_132" || true
fi

echo "slurm queue"
squeue -u "$USER" || true

echo "recent sacct"
sacct -u "$USER" -S "$(date -d '2 hours ago' +%Y-%m-%dT%H:%M:%S)" --format=JobID,JobName%28,State,ExitCode,Elapsed,AllocTRES%40 -P || true

echo "output files"
if [ -d "$OUTPUT_DIR" ]; then
  find "$OUTPUT_DIR" -maxdepth 3 -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %s %p\n" | sort | tail -n 80 || true
  echo "cell_count=$(find "$OUTPUT_DIR/cells" -maxdepth 1 -name 'cell_*.json' 2>/dev/null | wc -l | tr -d ' ')"
else
  echo "output_dir_missing"
fi

echo "inspect_helpsteer2_worker_quickdiag_task_done $(date)"
