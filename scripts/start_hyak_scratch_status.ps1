param(
  [string]$NetId = "lei0603",
  [string]$HostName = "klone.hyak.uw.edu",
  [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "HYAK SCRATCH STATUS - Codex"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "artifacts\hyak"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if ([string]::IsNullOrWhiteSpace($LogPath)) {
  $LogPath = Join-Path $LogDir "hyak_status.log"
}

$Target = "${NetId}@${HostName}"
$RemoteScript = @'
set -euo pipefail
RUN_REPO="/gscratch/scrubbed/$USER/ft-ppi/FT-PPI-runner"
echo "== HYAK SCRATCH STATUS =="
hostname
date
echo "run_repo=$RUN_REPO"
if [ ! -d "$RUN_REPO/.git" ]; then
  echo "scratch runner repo not found"
  exit 2
fi
cd "$RUN_REPO"
echo "repo=$(pwd)"
echo "head=$(git rev-parse --short HEAD 2>/dev/null || true)"
echo "== USER QUEUE =="
squeue -u "$USER" || true
echo "== RUNNER PROCESS =="
if [ -s .hyak_runner/runner.pid ]; then
  runner_pid="$(cat .hyak_runner/runner.pid 2>/dev/null || true)"
  echo "runner_pid=$runner_pid"
  if [ -n "$runner_pid" ]; then
    ps -fp "$runner_pid" || true
  fi
else
  echo "no runner pid file"
fi
echo "== TASK MARKERS =="
find .hyak_runner -maxdepth 2 -type f \( -path "*/running/*" -o -path "*/done/*" -o -path "*/failed/*" \) -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -30 || true
echo "== RECENT RUNNER OUTPUT =="
tail -200 .hyak_runner/runner.out 2>/dev/null || true
echo "== FOLLOW RUNNER OUTPUT =="
tail -n 0 -F .hyak_runner/runner.out
'@

Write-Host "Target: $Target"
Write-Host "Remote scratch repo: /gscratch/scrubbed/`$USER/ft-ppi/FT-PPI-runner"
Write-Host "Local log: $LogPath"
Write-Host ""
Write-Host "Enter UW password and complete Duo if prompted."
Write-Host "This status window will not restart the runner or submit jobs."
Write-Host ""

$EncodedRemoteScript = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($RemoteScript))
$RemoteCommand = "printf '%s' '$EncodedRemoteScript' | base64 -d | bash"
$SshExe = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
if (-not (Test-Path $SshExe)) {
  $SshExe = "ssh"
}

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  & $SshExe -n -o ServerAliveInterval=60 -o ServerAliveCountMax=10 $Target $RemoteCommand 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath $LogPath -Append
  $ExitCode = $LASTEXITCODE
} finally {
  $ErrorActionPreference = $PreviousErrorActionPreference
}

Write-Host ""
if ($ExitCode -and $ExitCode -ne 0) {
  Write-Host "Hyak scratch status command exited with code $ExitCode."
} else {
  Write-Host "Hyak scratch status command finished."
}
Write-Host "Press Enter to close this window."
Read-Host | Out-Null
