param(
    [string]$CondaBat = (Join-Path $env:USERPROFILE "anaconda3\condabin\conda.bat"),
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
    if ($LASTEXITCODE -ne 0) { throw "Failed to list Conda environments." }
    if ($envList -match "^\s*$EnvName\s+") {
        Write-Host "Environment $EnvName already exists. Skipping conda creation."
    }
    else {
        & $CondaBat --no-plugins env create --solver=classic -f environment.yml -n $EnvName
        if ($LASTEXITCODE -ne 0) { throw "Failed to create Conda environment $EnvName." }
    }
    & $CondaBat run -n $EnvName python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
    # Legacy NES/Gym and AutoROM builds need these packages available at build time.
    & $CondaBat run -n $EnvName python -m pip install setuptools==75.8.0 wheel==0.45.1 numpy==1.26.4 click requests tqdm==4.66.5
    if ($LASTEXITCODE -ne 0) { throw "Failed to install build dependencies." }
    & $CondaBat run -n $EnvName python -m pip install -r requirements.txt --no-build-isolation
    if ($LASTEXITCODE -ne 0) { throw "Failed to install dependencies." }
    & $CondaBat run -n $EnvName python -m pip install -e .
    if ($LASTEXITCODE -ne 0) { throw "Failed to install mario-rl." }
    Write-Host "Environment $EnvName is ready."
    Write-Host "Activate it with: conda activate $EnvName"
}
finally {
    Pop-Location
}
