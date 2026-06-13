#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

module purge
module load cuda/12.4.1
module load foster/python/miniconda/3.8 2>/dev/null || \
  module load chem/miniconda3/3.8 2>/dev/null || \
  module load coenv/miniconda/3.13.11 2>/dev/null || \
  module load coenv/python/3.11.9

choose_venv_dir() {
  if [ -n "${HYAK_VENV_DIR:-}" ]; then
    echo "$HYAK_VENV_DIR"
    return
  fi

  for base in \
    "${SCRATCH:-}" \
    "/gscratch/scrubbed/$USER" \
    "/gscratch/foster/$USER" \
    "$HOME/ft_ppi_envs"
  do
    [ -n "$base" ] || continue
    if mkdir -p "$base/ft-ppi" 2>/dev/null && [ -w "$base/ft-ppi" ]; then
      echo "$base/ft-ppi/.venv-hyak"
      return
    fi
  done

  echo ".venv-hyak"
}

VENV_DIR="$(choose_venv_dir)"
if [ "$VENV_DIR" != ".venv-hyak" ]; then
  if [ -e ".venv-hyak" ] && [ ! -L ".venv-hyak" ]; then
    mv ".venv-hyak" ".venv-hyak.home.old.$(date +%Y%m%d_%H%M%S)"
  fi
  ln -sfn "$VENV_DIR" ".venv-hyak"
fi
mkdir -p "$(dirname "$VENV_DIR")"
echo "VENV_DIR=$VENV_DIR"

CACHE_ROOT="${HYAK_CACHE_DIR:-$(dirname "$VENV_DIR")/cache}"
mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/hf_datasets" "$CACHE_ROOT/torch" "$CACHE_ROOT/pip" "$CACHE_ROOT/conda_pkgs"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/hf_datasets}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$CACHE_ROOT/conda_pkgs}"
echo "HF_HOME=$HF_HOME"

python3 --version
command -v conda >/dev/null 2>&1 && conda --version || true

if [ "${RESET:-0}" = "1" ] && [ -d "$VENV_DIR" ]; then
  backup="${VENV_DIR}.old.$(date +%Y%m%d_%H%M%S)"
  mv "$VENV_DIR" "$backup"
  echo "Moved existing environment to $backup"
fi

if [ -x "$VENV_DIR/bin/python" ] && ! "$VENV_DIR/bin/python" - <<'PY'
import _ctypes
PY
then
  backup="${VENV_DIR}.broken.$(date +%Y%m%d_%H%M%S)"
  mv "$VENV_DIR" "$backup"
  echo "Moved broken environment missing _ctypes to $backup"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  if command -v conda >/dev/null 2>&1; then
    conda create -y -p "$VENV_DIR" python=3.11 pip
  else
    python3 -m venv "$VENV_DIR"
  fi
fi

if [ -d "$VENV_DIR/conda-meta" ] && command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$VENV_DIR"
else
  source "$VENV_DIR/bin/activate"
fi

check_env() {
  python - <<'PY'
import importlib.util
import sys

import _ctypes

mods = ["torch", "transformers", "datasets", "accelerate", "peft", "bitsandbytes", "pandas", "pyarrow"]
missing = [mod for mod in mods if importlib.util.find_spec(mod) is None]
if missing:
    print("missing=" + ",".join(missing))
    sys.exit(1)

import torch
import pandas as pd
print("torch", torch.__version__)
print("cuda_build", torch.version.cuda)
print("pandas", pd.__version__)
PY
}

if [ "${FORCE_INSTALL:-0}" != "1" ] && check_env; then
  python -m pytest tests -q
  echo "Hyak environment ready. Activate with: source $VENV_DIR/bin/activate"
  exit 0
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip cache purge || true
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 "torch==2.5.1+cu124"
python -m pip install --no-cache-dir --no-deps accelerate==1.2.1 peft==0.14.0 bitsandbytes==0.45.1

python -m pytest tests -q

check_env

echo "Hyak environment ready. Activate with: source $VENV_DIR/bin/activate"
