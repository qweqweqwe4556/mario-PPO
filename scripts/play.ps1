param(
    [string]$CondaBat = "D:\conda3\condabin\conda.bat",
    [string]$EnvName = "mario-rl",
    [string]$ModelPath = "",
    [string]$Movement = "run-right",
    [int]$Episodes = 3
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

$ArgsList = @("-m", "mario_rl.play", "--movement", $Movement, "--episodes", "$Episodes")
if ($ModelPath -ne "") {
    $ArgsList += @("--model-path", $ModelPath)
}

Push-Location $Root
try {
    & $CondaBat run -n $EnvName python @ArgsList
}
finally {
    Pop-Location
}
