@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_prompting_pilot_local.ps1"
endlocal
