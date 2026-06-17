param(
    [string]$Python = "D:\Anaconda3\python.exe",
    [string]$Config = "configs\wine_full_grid_allocation.yaml",
    [string]$OutputDir = "artifacts\prompting_pilot_gpt55_n200",
    [int]$N = 200,
    [string]$Model = "gpt-5.5",
    [int]$BootstrapReps = 2000
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$LogPath = Join-Path $OutputDir "prompting_pilot_run.log"
Start-Transcript -Path $LogPath -Append | Out-Null
Write-Host "Prompting pilot log: $LogPath"

if (-not $env:OPENAI_API_KEY) {
    $secure = Read-Host "OpenAI API key" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        $env:OPENAI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

try {
    & $Python -m src.experiments.wine_prompting_pilot `
        --config $Config `
        --output-dir $OutputDir `
        --n $N `
        --model $Model `
        --bootstrap-reps $BootstrapReps

    if ($LASTEXITCODE -ne 0) {
        throw "Prompting pilot failed with exit code $LASTEXITCODE"
    }
}
finally {
    Stop-Transcript | Out-Null
}
