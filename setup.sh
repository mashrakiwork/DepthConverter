#!/usr/bin/env bash
# DepthConverter setup - Linux / macOS
# Installs uv (if missing), a managed Python, and all dependencies.
set -euo pipefail

echo "=== DepthConverter setup ==="

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$(dirname "$0")"
echo "Syncing environment (first run downloads PyTorch, this can take a while)..."
uv sync

echo
echo "Setup complete!"
echo "Start the app with:  uv run depthconverter"
echo "(Optional) Install a full ffmpeg for NVENC GPU encoding, e.g.: sudo apt install ffmpeg"
