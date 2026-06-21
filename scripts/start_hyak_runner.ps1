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
  $RemoteCommand = "cd $RemoteRepo && if git status --porcelain --untracked-files=all -- Data/wine_data.csv 2>/dev/null | grep -q '^?? '; then rm -f Data/wine_data.csv && echo removed untracked Data/wine_data.csv before pull; fi && git pull --ff-only origin $Branch && bash scripts/start_hyak_runner_remote.sh $Branch $PollSeconds $OnceValue"
} else {
  $RemoteCommand = "cd $RemoteRepo && if git status --porcelain --untracked-files=all -- Data/wine_data.csv 2>/dev/null | grep -q '^?? '; then rm -f Data/wine_data.csv && echo removed untracked Data/wine_data.csv before pull; fi && git pull --ff-only origin $Branch && HYAK_RUNNER_BRANCH=$Branch HYAK_RUNNER_POLL_SECONDS=$PollSeconds HYAK_RUNNER_ONCE=$OnceValue bash scripts/hyak_runner.sh"
}
$SshExe = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
if (-not (Test-Path $SshExe)) {
  throw "Windows OpenSSH not found at $SshExe. Refusing to use PATH ssh because Codex sandbox can shadow it."
}

Write-Host "Target: $Target"
Write-Host "Remote repo: $RemoteRepo"
Write-Host "Branch: $Branch"
Write-Host "Poll seconds: $PollSeconds"
Write-Host "Mode: $(if ($UseDetached) { 'detached remote runner with local tail' } else { 'foreground SSH runner' })"
Write-Host "Local log: $LogPath"
Write-Host "SSH executable: $SshExe"
Write-Host ""
Write-Host "Enter UW password and complete Duo if prompted."
if ($UseDetached) {
  Write-Host "After login, the remote runner will keep polling even if this window disconnects."
} else {
  Write-Host "Keep this window open while Codex is using Hyak."
}
Write-Host ""

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  & $SshExe -o ServerAliveInterval=60 -o ServerAliveCountMax=10 $Target $RemoteCommand 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath $LogPath -Append
  $ExitCode = $LASTEXITCODE
} finally {
  $ErrorActionPreference = $PreviousErrorActionPreference
}

Write-Host ""
if ($ExitCode -and $ExitCode -ne 0) {
  Write-Host "Hyak runner command exited with code $ExitCode."
} else {
  Write-Host "Hyak runner command finished."
}
Write-Host "Press Enter to close this window."
Read-Host | Out-Null
