param(
  [string]$NetId = "lei0603",
  [string]$HostName = "klone.hyak.uw.edu",
  [Parameter(Mandatory = $true)]
  [string]$Command
)

$ErrorActionPreference = "Stop"

$SshExe = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
if (-not (Test-Path $SshExe)) {
  throw "Windows OpenSSH not found at $SshExe."
}

$Target = "${NetId}@${HostName}"
$SafeHost = $HostName -replace '[^A-Za-z0-9_-]', '_'
$ControlPath = (($env:TEMP -replace "\\", "/") + "/ftppi_hyak_${NetId}_${SafeHost}_22_mux")

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$checkOutput = & $SshExe -S $ControlPath -O check $Target 2>&1
$checkExit = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($checkExit -ne 0) {
  [Console]::Error.WriteLine("No active Hyak SSH master at $ControlPath. Start scripts\start_hyak_runner.bat once and complete UW password/Duo first. Details: $checkOutput")
  exit 2
}

& $SshExe `
  -o ControlMaster=auto `
  -o "ControlPath=$ControlPath" `
  -o BatchMode=yes `
  -o ServerAliveInterval=60 `
  -o ServerAliveCountMax=10 `
  $Target `
  $Command
exit $LASTEXITCODE
