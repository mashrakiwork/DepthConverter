# DepthConverter setup - Windows
# Installs uv (if missing), a managed Python, all dependencies (CUDA torch
# included), and a full FFmpeg build (x265 + NVENC) into tools\ffmpeg.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # PS5.1: progress rendering slows downloads massively

Write-Host "=== DepthConverter setup ===" -ForegroundColor Cyan
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Python package manager)..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Syncing environment (first run downloads CUDA PyTorch, ~3 GB)..."
uv sync --extra da3
if ($LASTEXITCODE -ne 0) {
    Write-Host "Depth Anything V3 extra failed to install; continuing without it (V2/DPT still work)." -ForegroundColor Yellow
    uv sync
}

$ffdir = Join-Path $PSScriptRoot "tools\ffmpeg"
if (-not (Test-Path (Join-Path $ffdir "ffmpeg.exe"))) {
    Write-Host "Downloading full FFmpeg build (x265 + NVENC, ~180 MB)..."
    $zip = Join-Path $env:TEMP "depthconverter-ffmpeg.zip"
    $tmp = Join-Path $env:TEMP "depthconverter-ffmpeg-extract"
    Invoke-WebRequest -Uri "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip" -OutFile $zip
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    Expand-Archive -Path $zip -DestinationPath $tmp
    New-Item -ItemType Directory -Force $ffdir | Out-Null
    $exe = Get-ChildItem -Path $tmp -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
    Copy-Item (Join-Path $exe.DirectoryName "*") $ffdir -Force
    Remove-Item $zip -Force
    Remove-Item -Recurse -Force $tmp
    Write-Host "FFmpeg installed to tools\ffmpeg"
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Start the app by double-clicking DepthConverter.bat (or:  uv run depthconverter)"
Write-Host "Or pin DepthConverter.lnk - same app, no console flash at all."
