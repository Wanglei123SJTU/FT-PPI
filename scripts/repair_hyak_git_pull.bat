@echo off
setlocal
cd /d "%~dp0\.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0repair_hyak_git_pull.ps1"
echo.
echo Repair launcher finished. Press any key to close this window.
pause >nul
