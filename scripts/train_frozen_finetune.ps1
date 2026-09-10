param(
    [string]$CondaBat = (Join-Path $env:USERPROFILE "anaconda3\condabin\conda.bat"),
    [string]$EnvName = "mario-rl",
    [int]$TotalTimesteps = 8192,
    [int]$Seed = 43,
    [string]$Device = "cuda",
    [string]$LoadModel = "models\release\full_level_deterministic_baseline.zip",
    [string]$ModelDir = "models\frozen_finetune"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

$ArgsList = @(
    "-m", "mario_rl.train",
    "--load-model", $LoadModel,
    "--reset-optimizer",
    "--freeze-features",
    "--total-timesteps", "$TotalTimesteps",
    "--movement", "right",
    "--device", $Device,
    "--n-envs", "8",
    "--n-steps", "512",
    "--batch-size", "64",
    "--n-epochs", "3",
    "--seed", "$Seed",
    "--model-dir", $ModelDir,
    "--tb-log-name", "freeze_features_ne3_seed$Seed",
    "--max-jump-hold", "7",
    "--stuck-limit", "180",
    "--stuck-penalty", "25",
    "--death-penalty", "150",
    "--milestone-x", "2050",
    "--milestone-bonus", "250",
    "--progress-reward-scale", "0",
    "--flag-reward", "1000",
    "--learning-rate", "0.000005",
    "--clip-range", "0.1",
    "--ent-coef", "0.003",
    "--target-kl", "0.025",
    "--checkpoint-every", "$TotalTimesteps",
    "--xpos-eval-every", "0"
)

Push-Location $Root
try {
    & $CondaBat run -n $EnvName python @ArgsList
}
finally {
    Pop-Location
}
