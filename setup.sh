#!/usr/bin/env bash
# DepthConverter setup - Linux / macOS
# Installs uv (if missing), a managed Python, all dependencies, and (on Linux
# x86_64, when no system ffmpeg exists) a full FFmpeg build into tools/ffmpeg.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== DepthConverter setup ==="

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Syncing environment (first run downloads PyTorch, this can take a while)..."
if ! uv sync --extra da3; then
    echo "NOTE: Depth Anything V3 extra failed to install; continuing without it (V2/DPT still work)."
    uv sync
fi

FFDIR="$(pwd)/tools/ffmpeg"
if [ ! -x "$FFDIR/ffmpeg" ] && ! command -v ffmpeg >/dev/null 2>&1; then
    if [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
        echo "Downloading full FFmpeg build (x265 + NVENC)..."
        mkdir -p "$FFDIR"
        tmp=$(mktemp -d)
        curl -L -o "$tmp/ffmpeg.tar.xz" \
            "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-linux64-gpl.tar.xz"
        tar -xJf "$tmp/ffmpeg.tar.xz" -C "$tmp"
        cp "$tmp"/ffmpeg-*/bin/* "$FFDIR"/
        rm -rf "$tmp"
        echo "FFmpeg installed to tools/ffmpeg"
    else
        echo "NOTE: no ffmpeg found; install one via your package manager"
        echo "      (e.g. sudo apt install ffmpeg / brew install ffmpeg)."
        echo "      A bundled basic ffmpeg is used as fallback meanwhile."
    fi
fi

echo
echo "Setup complete!"
echo "Start the app with:  ./depthconverter.sh"
