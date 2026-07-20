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

# Create a console-free shortcut with the app icon (first run only). Launching
# via this shortcut - or pinning it - never opens a terminal window at all.
$lnk = Join-Path $PSScriptRoot "DepthConverter.lnk"
if (-not (Test-Path $lnk)) {
    $shell = New-Object -ComObject WScript.Shell
    $sc = $shell.CreateShortcut($lnk)
    $sc.TargetPath = $pythonw
    $sc.Arguments = "-m app.main"
    $sc.WorkingDirectory = $PSScriptRoot
    $sc.IconLocation = Join-Path $PSScriptRoot "app\assets\icon.ico"
    $sc.Description = "DepthConverter - local 2D to 3D VR"
    $sc.Save()
    Write-Host "Created DepthConverter.lnk - use it (or pin it) to launch with no console window."
}

Start-Process -FilePath $pythonw -ArgumentList "-m", "app.main" -WorkingDirectory $PSScriptRoot
