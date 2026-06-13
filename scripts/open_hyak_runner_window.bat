@echo off
cd /d "%~dp0\.."
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NoExit -ExecutionPolicy Bypass -File "%~dp0start_hyak_runner.ps1" %*
if errorlevel 1 (
  echo.
  echo Hyak runner launcher exited with error %errorlevel%.
  pause
)
