#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "Missing .venv. Run ./setup_env.sh first."
    exit 1
fi

PYTHONPATH="vendor${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python OX_Data_GUI.py
