#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

module purge
module load coenv/python/3.11.9
module load cuda/12.4.1

VENV_DIR="${HYAK_VENV_DIR:-.venv-hyak}"

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

python -m pip install --upgrade pip setuptools wheel
python -m pip cache purge || true
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 "torch==2.5.1+cu124"
python -m pip install --no-cache-dir --no-deps accelerate==1.2.1 peft==0.14.0 bitsandbytes==0.45.1

python -m pytest tests -q

python - <<'PY'
import importlib.util
mods = ["torch", "transformers", "datasets", "peft", "pandas", "pyarrow"]
for mod in mods:
    print(f"{mod}={importlib.util.find_spec(mod) is not None}")
PY

echo "Hyak environment ready. Activate with: source $VENV_DIR/bin/activate"
