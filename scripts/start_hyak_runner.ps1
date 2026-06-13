param(
  [string]$NetId = "lei0603",
  [string]$HostName = "klone.hyak.uw.edu",
  [string]$RemoteRepo = "~/FT-PPI",
  [string]$Branch = "main",
  [int]$PollSeconds = 60,
  [switch]$Once,
  [switch]$Foreground,
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
$UseDetached = -not $Foreground -and -not $Once
if ($UseDetached) {
  $RemoteCommand = "set -e; cd $RemoteRepo; git pull --ff-only origin $Branch; mkdir -p .hyak_runner; RUNNER_LOG=.hyak_runner/runner.out; RUNNER_PID=.hyak_runner/runner.pid; if [ -s `$RUNNER_PID ] && kill -0 `$(cat `$RUNNER_PID) 2>/dev/null; then echo detached runner already running pid=`$(cat `$RUNNER_PID); else echo starting detached runner; nohup env HYAK_RUNNER_BRANCH=$Branch HYAK_RUNNER_POLL_SECONDS=$PollSeconds HYAK_RUNNER_ONCE=$OnceValue bash scripts/hyak_runner.sh >> `$RUNNER_LOG 2>&1 < /dev/null & echo `$! > .hyak_runner/launcher.pid; fi; echo remote runner log: `$RUNNER_LOG; sleep 2; tail -n 120 -f `$RUNNER_LOG"
} else {
  $RemoteCommand = "cd $RemoteRepo && git pull --ff-only origin $Branch && HYAK_RUNNER_BRANCH=$Branch HYAK_RUNNER_POLL_SECONDS=$PollSeconds HYAK_RUNNER_ONCE=$OnceValue bash scripts/hyak_runner.sh"
}
$EscapedRemoteCommand = $RemoteCommand.Replace('"', '\"')
$CmdLine = "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 $Target ""$EscapedRemoteCommand"" 2>&1"

Write-Host "Target: $Target"
Write-Host "Remote repo: $RemoteRepo"
Write-Host "Branch: $Branch"
Write-Host "Poll seconds: $PollSeconds"
Write-Host "Mode: $(if ($UseDetached) { 'detached remote runner with local tail' } else { 'foreground SSH runner' })"
Write-Host "Local log: $LogPath"
Write-Host ""
Write-Host "Enter UW password and complete Duo if prompted."
if ($UseDetached) {
  Write-Host "After login, the remote runner will keep polling even if this window disconnects."
} else {
  Write-Host "Keep this window open while Codex is using Hyak."
}
Write-Host ""

cmd.exe /d /c $CmdLine | Tee-Object -FilePath $LogPath -Append

Write-Host ""
Write-Host "Hyak runner command finished."
