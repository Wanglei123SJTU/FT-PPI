param(
  [string]$NetId = "lei0603",
  [string]$HostName = "klone.hyak.uw.edu",
  [string]$Branch = "main",
  [int]$PollSeconds = 60,
  [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "HYAK SCRATCH RUNNER - Codex"

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
echo "== START HYAK SCRATCH RUNNER =="
hostname
date
SCRATCH_ROOT="/gscratch/scrubbed/$USER/ft-ppi"
RUN_REPO="$SCRATCH_ROOT/FT-PPI-runner"
mkdir -p "$SCRATCH_ROOT"
echo "scratch_root=$SCRATCH_ROOT"
echo "run_repo=$RUN_REPO"
if [ ! -d "$RUN_REPO/.git" ]; then
  rm -rf "$RUN_REPO"
  git clone --depth 1 --branch __BRANCH__ https://github.com/Wanglei123SJTU/FT-PPI.git "$RUN_REPO"
else
  cd "$RUN_REPO"
  git reset --hard HEAD
  if git status --porcelain --untracked-files=all -- Data/wine_data.csv 2>/dev/null | grep -q '^?? '; then
    rm -f Data/wine_data.csv
    echo "removed untracked Data/wine_data.csv before checkout"
  fi
  git fetch --depth 1 origin +__BRANCH__:refs/remotes/origin/__BRANCH__
  git checkout -B __BRANCH__ origin/__BRANCH__
fi
cd "$RUN_REPO"
echo "repo=$(pwd)"
echo "head=$(git rev-parse --short HEAD)"
if [ ! -f Data/wine_data.csv ] && [ -f Code/wine_data.csv ]; then
  mkdir -p Data
  ln -sfn ../Code/wine_data.csv Data/wine_data.csv
  echo "created Data/wine_data.csv symlink to Code/wine_data.csv"
fi
if [ -s "$HOME/FT-PPI/.hyak_runner/runner.pid" ]; then
  old_pid="$(cat "$HOME/FT-PPI/.hyak_runner/runner.pid" 2>/dev/null || true)"
  if [ -n "$old_pid" ]; then
    kill "$old_pid" 2>/dev/null || true
    echo "stopped home runner pid=$old_pid"
  fi
fi
if [ -s "$RUN_REPO/.hyak_runner/runner.pid" ]; then
  old_pid="$(cat "$RUN_REPO/.hyak_runner/runner.pid" 2>/dev/null || true)"
  if [ -n "$old_pid" ]; then
    kill "$old_pid" 2>/dev/null || true
    echo "stopped scratch runner pid=$old_pid"
  fi
fi
old_wine_var_jobs="$(squeue -u "$USER" -n wine-var -h -o "%A" 2>/dev/null | sort -u || true)"
if [ -n "$old_wine_var_jobs" ]; then
  echo "$old_wine_var_jobs" | xargs -r scancel
  echo "cancelled old wine-var jobs: $old_wine_var_jobs"
fi
mkdir -p .hyak_runner/done .hyak_runner/failed .hyak_runner/running
rm -f .hyak_runner/running/*
for task in hyak_tasks/*.sh; do
  task_id="$(basename "$task" .sh)"
  if [[ "$task_id" < "072_" ]]; then
    {
      echo "task_id=$task_id"
      echo "status=skipped_by_scratch_launcher"
      echo "finished_at=$(date +"%Y-%m-%dT%H:%M:%S%z")"
      echo "commit=$(git rev-parse --short HEAD)"
    } > ".hyak_runner/done/$task_id"
  fi
done
echo "marked older task files before 072 as skipped"
export HYAK_RUNNER_REPO_DIR="$RUN_REPO"
bash scripts/start_hyak_runner_remote.sh __BRANCH__ __POLL_SECONDS__ 0
'@
$RemoteScript = $RemoteScript.Replace("__BRANCH__", $Branch)
$RemoteScript = $RemoteScript.Replace("__POLL_SECONDS__", [string]$PollSeconds)

Write-Host "Target: $Target"
Write-Host "Remote scratch repo: /gscratch/scrubbed/`$USER/ft-ppi/FT-PPI-runner"
Write-Host "Branch: $Branch"
Write-Host "Local log: $LogPath"

$EncodedRemoteScript = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($RemoteScript))
$RemoteCommand = "printf '%s' '$EncodedRemoteScript' | base64 -d | bash"
$SshExe = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
if (-not (Test-Path $SshExe)) {
  throw "Windows OpenSSH not found at $SshExe. Refusing to use PATH ssh because Codex sandbox can shadow it."
}
Write-Host "SSH executable: $SshExe"
Write-Host ""
Write-Host "Enter UW password and complete Duo if prompted."
Write-Host "After login, this will clone/update the scratch repo and tail the detached runner."
Write-Host ""

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
  Write-Host "Hyak scratch runner command exited with code $ExitCode."
} else {
  Write-Host "Hyak scratch runner command finished."
}
Write-Host "Press Enter to close this window."
Read-Host | Out-Null
