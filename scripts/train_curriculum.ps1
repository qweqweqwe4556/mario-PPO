param(
    [string]$CondaBat = (Join-Path $env:USERPROFILE "anaconda3\condabin\conda.bat"),
    [string]$EnvName = "mario-rl",
    [int]$TotalTimesteps = 50000,
    [string]$Device = "auto",
    [string]$LoadModel = "models\release\full_level_deterministic_baseline.zip",
    [string]$ModelDir = "models\curriculum_next",
    [ValidateRange(0, 8)]
    [int]$FrontierEnvs = 2
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

$LoadModelPath = if ([System.IO.Path]::IsPathRooted($LoadModel)) {
    $LoadModel
}
else {
    Join-Path $Root $LoadModel
}
if (-not (Test-Path -LiteralPath $LoadModelPath)) {
    throw "Warm-start model not found: $LoadModel. Extract the release model asset or pass -LoadModel."
}

$ArgsList = @(
    "-m", "mario_rl.train",
    "--total-timesteps", "$TotalTimesteps",
    "--movement", "right",
    "--device", $Device,
    "--n-envs", "8",
    "--n-steps", "512",
    "--batch-size", "64",
    "--model-dir", $ModelDir,
    "--tb-log-name", "ppo_curriculum_next",
    "--load-model", $LoadModel,
    "--reset-optimizer",
    "--frontier-actions", "configs\frontier_x2095.json",
    "--frontier-envs", "$FrontierEnvs",
    "--max-jump-hold", "7",
    "--stuck-limit", "180",
    "--stuck-penalty", "150",
    "--death-penalty", "150",
    "--milestone-x", "2300",
    "--milestone-bonus", "250",
    "--progress-reward-scale", "0",
    "--flag-reward", "1000",
    "--learning-rate", "0.00001",
    "--clip-range", "0.1",
    "--ent-coef", "0.003",
    "--target-kl", "0.025",
    "--checkpoint-every", "10000",
    "--xpos-eval-every", "10000",
    "--xpos-eval-episodes", "10",
    "--xpos-eval-both",
    "--xpos-eval-frontier",
    "--xpos-eval-metric", "flag"
)

Push-Location $Root
try {
    & $CondaBat run -n $EnvName python @ArgsList
}
finally {
    Pop-Location
}
