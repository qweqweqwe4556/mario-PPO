param(
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

$TrainingProcesses = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "mario_rl\.train" }

if ($TrainingProcesses) {
    Write-Host "Mario training is running:"
    $TrainingProcesses |
        Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine |
        Format-List
}
else {
    Write-Host "No mario_rl.train process is running."
}

if ($LogPath -eq "") {
    $LatestLog = Get-ChildItem -LiteralPath (Join-Path $Root "logs") -Filter "train_*.log" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($LatestLog) {
        $LogPath = $LatestLog.FullName
    }
}

if ($LogPath -ne "" -and (Test-Path -LiteralPath $LogPath)) {
    Write-Host ""
    Write-Host "Latest log: $LogPath"
    Get-Content -LiteralPath $LogPath -Tail 80
}

$LatestErr = Get-ChildItem -LiteralPath (Join-Path $Root "logs") -Filter "train_*.err.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($LatestErr -and $LatestErr.Length -gt 0) {
    Write-Host ""
    Write-Host "Latest error log: $($LatestErr.FullName)"
    Get-Content -LiteralPath $LatestErr.FullName -Tail 80
}
