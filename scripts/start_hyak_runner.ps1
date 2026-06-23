param(
  [string]$NetId = "lei0603",
  [string]$HostName = "klone.hyak.uw.edu",
  [string]$RemoteRepo = "~/FT-PPI",
  [string]$Branch = "main",
  [int]$PollSeconds = 60,
  [switch]$Once,
  [switch]$Foreground,
  [switch]$NoMux,
  [int]$ControlPersistHours = 8,
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

$SafeHost = $HostName -replace '[^A-Za-z0-9_-]', '_'
$ControlPath = (($env:TEMP -replace "\\", "/") + "/ftppi_hyak_${NetId}_${SafeHost}_22_mux")
$UseMux = -not $NoMux

Write-Host "Target: $Target"
Write-Host "Remote repo: $RemoteRepo"
Write-Host "Branch: $Branch"
Write-Host "Poll seconds: $PollSeconds"
Write-Host "Mode: $(if ($UseDetached) { 'detached remote runner with local tail' } else { 'foreground SSH runner' })"
Write-Host "Local log: $LogPath"
Write-Host "SSH executable: $SshExe"
if ($UseMux) {
  Write-Host "SSH ControlPath: $ControlPath"
  Write-Host "ControlPersist: ${ControlPersistHours}h"
} else {
  Write-Host "SSH mux disabled."
}
Write-Host ""
Write-Host "Enter UW password and complete Duo if prompted."
if ($UseDetached) {
  Write-Host "After login, the remote runner will keep polling even if this window disconnects."
} else {
  Write-Host "Keep this window open while Codex is using Hyak."
}
if ($UseMux) {
  Write-Host "This launcher will try to establish a reusable SSH master first. If Windows OpenSSH rejects mux, it will fall back to one normal SSH login and start the detached remote runner."
}
Write-Host ""

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  if ($UseMux) {
    $MuxAlive = $false
    if (Test-Path $ControlPath) {
      $MuxCheck = & $SshExe -S $ControlPath -O check $Target 2>&1
      $MuxAlive = ($LASTEXITCODE -eq 0)
    }
    if ($MuxAlive) {
      Write-Host "Reusing existing SSH master connection."
    } else {
      if (Test-Path $ControlPath) {
        Remove-Item -Force $ControlPath -ErrorAction SilentlyContinue
      }
      Write-Host "Starting SSH master connection. Authenticate once here."
      $MasterArgs = @(
        "-M",
        "-N",
        "-f",
        "-o", "ControlMaster=yes",
        "-o", "ControlPath=$ControlPath",
        "-o", "ControlPersist=${ControlPersistHours}h",
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=10",
        $Target
      )
      & $SshExe @MasterArgs
      if ($LASTEXITCODE -ne 0) {
        Write-Host "SSH master setup failed; falling back to one normal SSH connection for the detached runner."
        Write-Host "This is acceptable on Windows: the remote runner will continue polling after it starts."
        $UseMux = $false
      } else {
        Write-Host "SSH master established."
      }
    }
  }

  if ($UseMux) {
    $RemoteArgs = @(
      "-o", "ControlMaster=auto",
      "-o", "ControlPath=$ControlPath",
      "-o", "BatchMode=yes",
      "-o", "ServerAliveInterval=60",
      "-o", "ServerAliveCountMax=10",
      $Target,
      $RemoteCommand
    )
    & $SshExe @RemoteArgs 2>&1 |
      ForEach-Object { $_.ToString() } |
      Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE
  } else {
    Write-Host "Starting normal SSH runner connection. Authenticate once here if prompted."
    & $SshExe -o ServerAliveInterval=60 -o ServerAliveCountMax=10 $Target $RemoteCommand 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE
  }
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
