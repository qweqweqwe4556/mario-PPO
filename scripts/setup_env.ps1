param(
    [string]$CondaBat = "D:\conda3\condabin\conda.bat",
    [string]$EnvName = "mario-rl"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path -LiteralPath $CondaBat)) {
    throw "Cannot find conda at $CondaBat. Pass -CondaBat with the full path to conda.bat."
}

Push-Location $Root
try {
    $envList = & $CondaBat env list
    if ($envList -match "^\s*$EnvName\s+") {
        Write-Host "Environment $EnvName already exists. Skipping conda creation."
    }
    else {
        & $CondaBat --no-plugins env create --solver=classic -f environment.yml
    }
    & $CondaBat run -n $EnvName python -m pip install --upgrade pip
    & $CondaBat run -n $EnvName python -m pip install -r requirements.txt
    & $CondaBat run -n $EnvName python -m pip install -e .
    Write-Host "Environment $EnvName is ready."
    Write-Host "Activate it with: conda activate $EnvName"
}
finally {
    Pop-Location
}
