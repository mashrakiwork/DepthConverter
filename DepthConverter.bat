@echo off
rem The launcher: double-click this, or pin the DepthConverter shortcut it
rem creates on first run.
rem
rem pythonw.exe has no console of its own, so once this hands off there is no
rem terminal window attached to the app. The brief flash of this window while
rem it runs is the one thing a .bat cannot avoid - use the shortcut to skip
rem even that.
setlocal
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

if exist "%ROOT%\.venv\Scripts\pythonw.exe" if exist "%ROOT%\tools\ffmpeg\ffmpeg.exe" (
    start "" /D "%ROOT%" "%ROOT%\.venv\Scripts\pythonw.exe" -m app.main
    exit /b
)

rem First run - or the project folder was moved, which invalidates the venv's
rem absolute paths. Set up in view (it downloads several GB), then start. This
rem is the only path that shows a console, and it also rewrites the shortcut
rem for wherever the project now lives.
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\launch.ps1"
