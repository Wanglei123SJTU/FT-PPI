param(
  [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$ScriptPath = Resolve-Path (Join-Path $PSScriptRoot "start_hyak_runner.ps1")
$CommandLine = "powershell.exe -NoProfile -NoExit -ExecutionPolicy Bypass -File `"$ScriptPath`" -PollSeconds $PollSeconds"
Start-Process cmd.exe -ArgumentList "/k", $CommandLine
