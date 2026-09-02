@echo off
rem Hands straight over to the console-free VBS launcher, then exits - so no
rem terminal window is ever left attached to the app window.
start "" wscript.exe "%~dp0DepthConverter.vbs"
