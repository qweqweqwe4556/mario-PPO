param(
    [string]$CondaBat = "D:\conda3\condabin\conda.bat",
    [string]$EnvName = "mario-rl",
    [int]$TotalTimesteps = 100000,
    [string]$Movement = "run-right",
    [string]$Device = "auto",
    [int]$NEnvs = 1,
    [string]$ModelDir = "models",
    [string]$TbLogName = "ppo_mario"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Push-Location $Root
try {
    & $CondaBat run -n $EnvName python -m mario_rl.train `
        --total-timesteps $TotalTimesteps `
        --movement $Movement `
        --device $Device `
        --n-envs $NEnvs `
        --model-dir $ModelDir `
        --tb-log-name $TbLogName `
        --resume
}
finally {
    Pop-Location
}
