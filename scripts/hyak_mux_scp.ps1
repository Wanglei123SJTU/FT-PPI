param(
  [string]$NetId = "lei0603",
  [string]$HostName = "klone.hyak.uw.edu",
  [Parameter(Mandatory = $true)]
  [string[]]$LocalPath,
  [Parameter(Mandatory = $true)]
  [string]$RemotePath
)

$ErrorActionPreference = "Stop"

$SshExe = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
$ScpExe = Join-Path $env:SystemRoot "System32\OpenSSH\scp.exe"
if (-not (Test-Path $SshExe)) {
  throw "Windows OpenSSH ssh.exe not found at $SshExe."
}
if (-not (Test-Path $ScpExe)) {
  throw "Windows OpenSSH scp.exe not found at $ScpExe."
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

$args = @(
  "-o", "ControlMaster=auto",
  "-o", "ControlPath=$ControlPath",
  "-o", "BatchMode=yes",
  "-o", "ServerAliveInterval=60",
  "-o", "ServerAliveCountMax=10"
)
$args += $LocalPath
$args += "${Target}:$RemotePath"

& $ScpExe @args
exit $LASTEXITCODE
