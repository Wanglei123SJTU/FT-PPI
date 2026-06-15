@echo off
setlocal
title HYAK SCRATCH RUNNER - Codex
cd /d "%~dp0\.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_hyak_scratch_runner.ps1"
echo.
echo Scratch runner launcher finished. Press any key to close this window.
pause >nul
