param(
    [string]$CondaBat = "D:\conda3\condabin\conda.bat",
    [string]$EnvName = "mario-rl",
    [int]$TotalTimesteps = 500000,
    [string]$Movement = "right",
    [string]$Device = "cpu",
    [int]$NEnvs = 1,
    [string]$ModelDir = "models\pass_run",
    [string]$TbLogName = "ppo_mario_pass_run",
    [string]$LoadModel = "",
    [int]$StuckLimit = 120
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root "logs"
$LogFile = Join-Path $LogDir ("train_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
$ErrFile = Join-Path $LogDir ("train_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".err.log")

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Arguments = @(
    "/c",
    "set",
    "PYTHONUNBUFFERED=1",
    "&&",
    "call",
    "`"$CondaBat`"",
    "run",
    "-n",
    $EnvName,
    "python",
    "-m",
    "mario_rl.train",
    "--total-timesteps",
    "$TotalTimesteps",
    "--movement",
    $Movement,
    "--device",
    $Device,
    "--n-envs",
    "$NEnvs",
    "--model-dir",
    "`"$ModelDir`"",
    "--tb-log-name",
    $TbLogName,
    "--checkpoint-every",
    "25000",
    "--xpos-eval-every",
    "25000",
    "--xpos-eval-episodes",
    "3",
    "--stuck-limit",
    "$StuckLimit",
    "--learning-rate",
    "0.0001",
    "--clip-range",
    "0.2",
    "--ent-coef",
    "0.02",
    "--resume"
)

if ($LoadModel -ne "") {
    $Arguments += @("--load-model", "`"$LoadModel`"")
}

$Process = Start-Process `
    -FilePath "cmd.exe" `
    -ArgumentList $Arguments `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $LogFile `
    -RedirectStandardError $ErrFile `
    -PassThru

Write-Host "Started background training."
Write-Host "Process ID: $($Process.Id)"
Write-Host "Log file: $LogFile"
Write-Host "Error log: $ErrFile"
Write-Host "Model dir: $ModelDir"
