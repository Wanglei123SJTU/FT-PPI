@echo off
setlocal
title HYAK SCRATCH STATUS - Codex
cd /d "%~dp0\.."
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_hyak_scratch_status.ps1"
echo.
echo Hyak scratch status window finished. Press any key to close this window.
pause >nul
