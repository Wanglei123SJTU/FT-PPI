#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

module purge
module load coenv/python/3.11.9
module load cuda/12.4.1

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

python3 --version

if [ "${RESET:-0}" = "1" ] && [ -d "$VENV_DIR" ]; then
  backup="${VENV_DIR}.old.$(date +%Y%m%d_%H%M%S)"
  mv "$VENV_DIR" "$backup"
  echo "Moved existing environment to $backup"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

check_env() {
  python - <<'PY'
import importlib.util
import sys

mods = ["torch", "transformers", "datasets", "accelerate", "peft", "bitsandbytes", "pandas", "pyarrow"]
missing = [mod for mod in mods if importlib.util.find_spec(mod) is None]
if missing:
    print("missing=" + ",".join(missing))
    sys.exit(1)

import torch
print("torch", torch.__version__)
print("cuda_build", torch.version.cuda)
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
