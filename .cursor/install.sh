#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the RSNA knee MRI classifier scaffold.
# Runs after the repository is checked out. Creates a venv and installs the
# pinned Python dependencies from requirements.txt.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

# venv support is not guaranteed on the base image.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends python3-venv
fi

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

echo "install complete: $(python --version)"
