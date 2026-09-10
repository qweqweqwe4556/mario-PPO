param(
    [string]$CondaPython = "D:\anaconda\envs\mario-rl\python.exe",
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8765,
    [int]$WebPort = 5173
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$WebDir = Join-Path $Root "web"

if (-not (Test-Path -LiteralPath $CondaPython)) {
    throw "Python not found: $CondaPython"
}

if (-not (Test-Path -LiteralPath (Join-Path $WebDir "node_modules"))) {
    Write-Host "Installing frontend dependencies..."
    Push-Location $WebDir
    try {
        npm install
        if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
    }
    finally {
        Pop-Location
    }
}

Write-Host "Starting Mario PPO demo console..."
Write-Host "  API : http://$ApiHost`:$ApiPort"
Write-Host "  Web : http://127.0.0.1:$WebPort"
Write-Host "Keep both windows open during the presentation."

$api = Start-Process -FilePath $CondaPython -ArgumentList @(
    "-m", "uvicorn", "mario_rl.demo_api:app",
    "--host", $ApiHost,
    "--port", "$ApiPort"
) -WorkingDirectory $Root -PassThru -WindowStyle Normal

Start-Sleep -Seconds 2

Push-Location $WebDir
try {
    $env:BROWSER = "none"
    npm run dev
}
finally {
    Pop-Location
    if ($api -and -not $api.HasExited) {
        Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
    }
}
