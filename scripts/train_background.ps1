param(
    [string]$CondaBat = (Join-Path $env:USERPROFILE "anaconda3\condabin\conda.bat"),
    [string]$EnvName = "mario-rl",
    [int]$TotalTimesteps = 100000,
    [string]$Movement = "right",
    [string]$Device = "auto",
    [int]$NEnvs = 1,
    [string]$ModelDir = "models\background",
    [string]$TbLogName = "ppo_mario_background",
    [string]$LoadModel = "",
    [int]$StuckLimit = 120,
    [int]$MaxJumpHold = 0,
    [int]$CheckpointEvery = 25000,
    [int]$EvalEvery = 25000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root "logs"
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir ("train_" + $RunStamp + ".log")
$ErrFile = Join-Path $LogDir ("train_" + $RunStamp + ".err.log")

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
    "$CheckpointEvery",
    "--xpos-eval-every",
    "$EvalEvery",
    "--xpos-eval-episodes",
    "3",
    "--stuck-limit",
    "$StuckLimit",
    "--max-jump-hold",
    "$MaxJumpHold",
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
