#!/bin/bash
set -euo pipefail

echo "upworthy_feature_pilots_watch_task_start"
date
hostname
echo "commit=$(git rev-parse --short HEAD)"

ROOT="/gscratch/scrubbed/lei0603/ft-ppi"
OUTS=(
  "$ROOT/artifacts/upworthy_length_single_scaling_pilot_1p5b"
  "$ROOT/artifacts/upworthy_simplicity_single_scaling_pilot_1p5b"
  "$ROOT/artifacts/upworthy_length_full5_scaling_pilot_1p5b"
  "$ROOT/artifacts/upworthy_length_single_scaling_pilot_7b"
)

for round in 1 2 3 4 5; do
  echo "== watch round $round =="
  date

  echo "-- squeue --"
  squeue -u lei0603 || true

  echo "-- sacct recent feature-pilot jobs --"
  sacct -u lei0603 --starttime now-12hours --format=JobID,JobName%24,Partition,State,ExitCode,Elapsed,MaxRSS | grep -E "upworthy-qscale|361855|JobID" || true

  echo "-- output metric counts --"
  for out in "${OUTS[@]}"; do
    echo "--- $out ---"
    if [[ -d "$out" ]]; then
      echo "metrics_count=$(find "$out" -name metrics.json | wc -l)"
      find "$out" -maxdepth 1 -type f -printf "%f %s bytes\n" 2>/dev/null | sort || true
      if [[ -f "$out/aggregate_metrics.csv" ]]; then
        echo "aggregate_head"
        head -20 "$out/aggregate_metrics.csv" || true
      fi
      if [[ -f "$out/scaling_fits.csv" ]]; then
        echo "scaling_fits"
        cat "$out/scaling_fits.csv" || true
      fi
    else
      echo "missing"
    fi
  done

  echo "-- latest slurm logs --"
  for f in $(ls -1t "$ROOT/logs"/upworthy-qscale-*.out 2>/dev/null | head -8); do
    echo "--- tail $f ---"
    tail -50 "$f" || true
  done

  if [[ "$round" != "5" ]]; then
    sleep 120
  fi
done

echo "upworthy_feature_pilots_watch_task_done"
