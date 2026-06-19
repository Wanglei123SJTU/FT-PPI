#!/bin/bash
set -euo pipefail

echo "upworthy_feature_pilots_status_task_start"
date
hostname
echo "commit=$(git rev-parse --short HEAD)"

echo "== runner markers =="
for d in .hyak_runner/running .hyak_runner/done .hyak_runner/failed; do
  echo "-- $d --"
  ls -1 "$d" 2>/dev/null | tail -50 || true
done

echo "== squeue for lei0603 =="
squeue -u lei0603 || true

echo "== sacct recent upworthy jobs =="
sacct -u lei0603 --starttime now-12hours --format=JobID,JobName%24,Partition,State,ExitCode,Elapsed,MaxRSS | grep -E "upworthy|36185|JobID" || true

echo "== task 083 remote runner log tail =="
tail -220 .hyak_runner/logs/083_cancel_l40s_and_rerun_feature_target_pilots_ckptg2.log 2>/dev/null || true

echo "== task 082 remote runner log tail =="
tail -120 .hyak_runner/logs/082_upworthy_feature_target_pilots.log 2>/dev/null || true

echo "== selected slurm log tails =="
for f in $(ls -1t logs/upworthy-qscale-36185532_*.out 2>/dev/null | head -12); do
  echo "--- tail $f ---"
  tail -80 "$f" || true
done

echo "== output metric counts =="
for out in \
  artifacts/upworthy_length_single_scaling_pilot_1p5b \
  artifacts/upworthy_simplicity_single_scaling_pilot_1p5b \
  artifacts/upworthy_length_full5_scaling_pilot_1p5b \
  artifacts/upworthy_length_single_scaling_pilot_7b
do
  echo "-- $out --"
  if [[ -d "$out" ]]; then
    find "$out" -name metrics.json | wc -l
    find "$out" -maxdepth 1 -type f -printf "%f %s bytes\n" 2>/dev/null | sort || true
  else
    echo "missing"
  fi
done

echo "upworthy_feature_pilots_status_task_done"
