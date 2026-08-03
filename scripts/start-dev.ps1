param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Frontend = Join-Path $Root "frontend"

Set-Location $Root

if (!(Test-Path $Python)) {
    python -m venv .venv
}

if (!$SkipInstall) {
    & $Python -m pip install -r requirements.txt
    if (!(Test-Path (Join-Path $Frontend "node_modules"))) {
        Push-Location $Frontend
        npm.cmd install
        Pop-Location
    }
}

& $Python -m alembic upgrade head
& $Python data_pipeline.py --provider sample

$ApiJob = Start-Job -Name horse-predictor-api -ScriptBlock {
    param($RootPath, $PythonPath)
    Set-Location $RootPath
    & $PythonPath -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
} -ArgumentList $Root, $Python

$FrontendJob = Start-Job -Name horse-predictor-frontend -ScriptBlock {
    param($FrontendPath)
    Set-Location $FrontendPath
    npm.cmd run dev
} -ArgumentList $Frontend

Write-Host "Horse Predictor is starting."
Write-Host "Backend:  http://127.0.0.1:8000/api/health"
Write-Host "Frontend: http://127.0.0.1:5173"
Write-Host "Press Ctrl+C to stop both services."

try {
    while ($true) {
        Receive-Job -Job $ApiJob, $FrontendJob
        $failed = @($ApiJob, $FrontendJob) | Where-Object { $_.State -in @("Failed", "Stopped", "Completed") }
        if ($failed.Count -gt 0) {
            throw "A dev service stopped: $($failed.Name -join ', ')"
        }
        Start-Sleep -Seconds 2
    }
}
finally {
    Stop-Job -Job $ApiJob, $FrontendJob -ErrorAction SilentlyContinue
    Remove-Job -Job $ApiJob, $FrontendJob -Force -ErrorAction SilentlyContinue
}
