#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 was not found. Install Python 3 from python.org or Homebrew first."
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-portable.txt

echo
echo "Setup complete. You can now run run_gui.command."
read -r -p "Press Enter to close this window..."
