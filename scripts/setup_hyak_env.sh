#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

module purge
module load coenv/python/3.11.9
module load cuda/12.4.1

python3 --version
python3 -m venv --clear .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip cache purge || true
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 "torch==2.5.1+cu124"
python -m pip install --no-cache-dir -r requirements-hyak.txt

python -m pytest tests -q

python - <<'PY'
import importlib.util
mods = ["torch", "transformers", "datasets", "peft", "pandas", "pyarrow"]
for mod in mods:
    print(f"{mod}={importlib.util.find_spec(mod) is not None}")
PY

echo "Hyak environment ready. Activate with: source .venv/bin/activate"
