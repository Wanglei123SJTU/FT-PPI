#!/bin/bash
set -euo pipefail

echo "cancel_length_budget_sweep_task_start"
hostname
date
git rev-parse --short HEAD

echo "canceling Slurm job 36202188 if present"
scancel 36202188 || true
sleep 5

echo "squeue after cancel:"
squeue -u "$USER" -j 36202188 || true

echo "sacct after cancel:"
sacct -j 36202188 --format=JobID,State,ExitCode,Elapsed -X || true

echo "cancel_length_budget_sweep_task_done"
