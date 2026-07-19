# DepthConverter setup - Windows
# Installs uv (if missing), a managed Python, and all dependencies (CUDA torch included).
$ErrorActionPreference = "Stop"

Write-Host "=== DepthConverter setup ===" -ForegroundColor Cyan

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Python package manager)..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Syncing environment (first run downloads CUDA PyTorch, ~3 GB)..."
Set-Location $PSScriptRoot
uv sync

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Start the app with:  uv run depthconverter"
Write-Host "(Optional) Install a full ffmpeg for NVENC GPU encoding: winget install Gyan.FFmpeg"
