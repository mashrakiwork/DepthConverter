#!/usr/bin/env bash
# DepthConverter launcher (Linux/macOS) - runs setup on first use, then starts the app.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "First run - setting everything up (this can take a while)..."
    ./setup.sh
fi

exec .venv/bin/python -m app.main
