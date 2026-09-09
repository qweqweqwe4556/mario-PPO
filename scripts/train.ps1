param(
    [string]$CondaBat = (Join-Path $env:USERPROFILE "anaconda3\condabin\conda.bat"),
    [string]$EnvName = "mario-rl",
    [int]$TotalTimesteps = 100000,
    [string]$Movement = "right",
    [string]$Device = "auto",
    [int]$NEnvs = 1,
    [string]$ModelDir = "models",
    [string]$TbLogName = "ppo_mario",
    [int]$MaxJumpHold = 0,
    [string]$LoadModel = "",
    [int]$CheckpointEvery = 25000,
    [int]$EvalEvery = 25000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Push-Location $Root
try {
    $ArgsList = @(
        "-m", "mario_rl.train",
        "--total-timesteps", "$TotalTimesteps",
        "--movement", $Movement,
        "--device", $Device,
        "--n-envs", "$NEnvs",
        "--model-dir", $ModelDir,
        "--tb-log-name", $TbLogName,
        "--max-jump-hold", "$MaxJumpHold",
        "--checkpoint-every", "$CheckpointEvery",
        "--xpos-eval-every", "$EvalEvery",
        "--resume"
    )
    if ($LoadModel -ne "") {
        $ArgsList += @("--load-model", $LoadModel)
    }
    & $CondaBat run -n $EnvName python @ArgsList
}
finally {
    Pop-Location
}
