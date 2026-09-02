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

# Keep a console-free shortcut with the app icon pointing at THIS folder.
# Rewritten whenever it is missing or stale (e.g. the project folder moved), so
# a shortcut carried over from an old path can never end up launching nothing.
$lnk = Join-Path $PSScriptRoot "DepthConverter.lnk"
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnk)
if ($sc.TargetPath -ne $pythonw) {
    $sc.TargetPath = $pythonw
    $sc.Arguments = "-m app.main"
    $sc.WorkingDirectory = $PSScriptRoot
    $sc.IconLocation = (Join-Path $PSScriptRoot "app\assets\icon.ico") + ",0"
    $sc.Description = "DepthConverter - local 2D to 3D VR"
    $sc.Save()
    Write-Host "Wrote DepthConverter.lnk - use it (or pin it) to launch with no console window."
}

Start-Process -FilePath $pythonw -ArgumentList "-m", "app.main" -WorkingDirectory $PSScriptRoot
