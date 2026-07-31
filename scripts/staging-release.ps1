param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

if (!$env:DATABASE_URL) {
    throw "DATABASE_URL must be set before running the staging release."
}

& $Python -m alembic upgrade head

if ($env:SEED_SAMPLE_DATA -eq "true") {
    & $Python data_pipeline.py --provider sample --no-csv --disable-lock
}

Write-Host "Staging release checks completed."
