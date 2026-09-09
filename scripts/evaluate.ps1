param(
    [string]$CondaBat = (Join-Path $env:USERPROFILE "anaconda3\condabin\conda.bat"),
    [string]$EnvName = "mario-rl",
    [string]$ModelPath = "",
    [string]$Movement = "right",
    [int]$Episodes = 5,
    [string]$Device = "auto",
    [int]$MaxJumpHold = 0,
    [string]$FrontierActions = "",
    [switch]$Stochastic
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

$ArgsList = @(
    "-m", "mario_rl.evaluate",
    "--movement", $Movement,
    "--episodes", "$Episodes",
    "--device", $Device,
    "--max-jump-hold", "$MaxJumpHold"
)
if ($ModelPath -ne "") {
    $ArgsList += @("--model-path", $ModelPath)
}
if ($FrontierActions -ne "") {
    $ArgsList += @("--frontier-actions", $FrontierActions)
}
if ($Stochastic) {
    $ArgsList += "--stochastic"
}

Push-Location $Root
try {
    & $CondaBat run -n $EnvName python @ArgsList
}
finally {
    Pop-Location
}
