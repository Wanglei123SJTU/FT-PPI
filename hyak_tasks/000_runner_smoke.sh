#!/bin/bash
set -euo pipefail

echo "runner_smoke_task_start"
echo "task_id=${HYAK_RUNNER_TASK_ID:-unknown}"
hostname
date
git rev-parse --short HEAD

if [ ! -x .venv-hyak/bin/python ]; then
  echo "missing .venv-hyak/bin/python"
  exit 1
fi

.venv-hyak/bin/python - <<'PY'
import _ctypes
import pandas as pd
import torch

print("python_env_ok")
print("torch", torch.__version__)
print("cuda_build", torch.version.cuda)
print("pandas", pd.__version__)
PY

echo "runner_smoke_task_done"
