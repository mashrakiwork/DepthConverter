# DepthConverter launcher - runs setup automatically on first use, then starts
# the app without keeping a console window open.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$pythonw = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
$ffmpeg = Join-Path $PSScriptRoot "tools\ffmpeg\ffmpeg.exe"

if (-not (Test-Path $pythonw) -or -not (Test-Path $ffmpeg)) {
    Write-Host "First run - setting everything up (this can take a while)..."
    & (Join-Path $PSScriptRoot "setup.ps1")
}

Start-Process -FilePath $pythonw -ArgumentList "-m", "app.main" -WorkingDirectory $PSScriptRoot
