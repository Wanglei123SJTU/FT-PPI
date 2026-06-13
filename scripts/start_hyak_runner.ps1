param(
  [string]$NetId = "lei0603",
  [string]$HostName = "klone.hyak.uw.edu",
  [string]$RemoteRepo = "~/FT-PPI",
  [string]$Branch = "main",
  [int]$PollSeconds = 60,
  [switch]$Once,
  [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "HYAK RUNNER - Codex"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "artifacts\hyak"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if ([string]::IsNullOrWhiteSpace($LogPath)) {
  $LogPath = Join-Path $LogDir "hyak_runner.log"
}

$Target = "${NetId}@${HostName}"
$OnceValue = if ($Once) { "1" } else { "0" }
$RemoteScript = @"
set -euo pipefail
cd $RemoteRepo
git pull --ff-only origin $Branch
export HYAK_RUNNER_BRANCH="$Branch"
export HYAK_RUNNER_POLL_SECONDS="$PollSeconds"
export HYAK_RUNNER_ONCE="$OnceValue"
exec bash scripts/hyak_runner.sh
"@ -replace "`r`n", "`n"

Write-Host "Target: $Target"
Write-Host "Remote repo: $RemoteRepo"
Write-Host "Branch: $Branch"
Write-Host "Poll seconds: $PollSeconds"
Write-Host "Local log: $LogPath"
Write-Host ""
Write-Host "Enter UW password and complete Duo if prompted."
Write-Host "Keep this window open while Codex is using Hyak."
Write-Host ""

$RemoteScript | ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 $Target 'bash -s' 2>&1 |
  Tee-Object -FilePath $LogPath -Append

Write-Host ""
Write-Host "Hyak runner command finished."
