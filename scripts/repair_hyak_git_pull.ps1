param(
  [string]$NetId = "lei0603",
  [string]$HostName = "klone.hyak.uw.edu",
  [string]$RemoteRepo = "~/FT-PPI",
  [string]$Branch = "main",
  [int]$PollSeconds = 60,
  [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "HYAK REPAIR + RUNNER - Codex"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "artifacts\hyak"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if ([string]::IsNullOrWhiteSpace($LogPath)) {
  $LogPath = Join-Path $LogDir "hyak_runner.log"
}

$Target = "${NetId}@${HostName}"
$RemoteScript = @'
set -euo pipefail
echo "== REPAIR HYAK GIT PULL =="
hostname
date
cd __REMOTE_REPO__
mkdir -p .hyak_runner/logs
echo "repo=\$(pwd)"
echo "before_head=\$(git rev-parse --short HEAD 2>/dev/null || true)"
echo "before_git_size=\$(du -sh .git 2>/dev/null || true)"
echo "== remove failed fetch temp files =="
find .git/objects -type f \( -name 'tmp_*' -o -name 'tmp_pack_*' \) -print -delete 2>/dev/null || true
find .git/objects -type d \( -name 'tmp_objdir-*' -o -name 'incoming-*' \) -print -exec rm -rf {} + 2>/dev/null || true
rm -f .git/index.lock .git/packed-refs.lock .git/refs/remotes/origin/main.lock .git/FETCH_HEAD.lock 2>/dev/null || true
git reflog expire --expire=now --all 2>/dev/null || true
git gc --prune=now 2>/dev/null || true
echo "after_git_size=\$(du -sh .git 2>/dev/null || true)"
echo "== fetch forced origin/main after local history repair =="
git fetch --prune origin +__BRANCH__:refs/remotes/origin/__BRANCH__
git merge --ff-only origin/__BRANCH__
echo "after_head=\$(git rev-parse --short HEAD)"
echo "== start detached runner =="
bash scripts/start_hyak_runner_remote.sh __BRANCH__ __POLL_SECONDS__ 0
'@
$RemoteScript = $RemoteScript.Replace("__REMOTE_REPO__", $RemoteRepo)
$RemoteScript = $RemoteScript.Replace("__BRANCH__", $Branch)
$RemoteScript = $RemoteScript.Replace("__POLL_SECONDS__", [string]$PollSeconds)

Write-Host "Target: $Target"
Write-Host "Remote repo: $RemoteRepo"
Write-Host "Branch: $Branch"
Write-Host "Local log: $LogPath"
Write-Host ""
Write-Host "Enter UW password and complete Duo if prompted."
Write-Host "This will repair the remote Git pull, then tail the detached Hyak runner."
Write-Host ""

$EncodedRemoteScript = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($RemoteScript))
$RemoteCommand = "printf '%s' '$EncodedRemoteScript' | base64 -d | bash"
$SshExe = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
if (-not (Test-Path $SshExe)) {
  throw "Windows OpenSSH not found at $SshExe. Refusing to use PATH ssh because Codex sandbox can shadow it."
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
  Write-Host "Hyak repair command exited with code $ExitCode."
} else {
  Write-Host "Hyak repair command finished."
}
Write-Host "Press Enter to close this window."
Read-Host | Out-Null
