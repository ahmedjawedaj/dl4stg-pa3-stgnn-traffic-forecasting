#!/usr/bin/env bash
# Creates .venv with the CPU dependencies.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo "Activate with: source .venv/bin/activate"
