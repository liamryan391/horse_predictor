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

if ($env:RUN_ACCEPTANCE_CHECKS -eq "true") {
    if (!$env:API_BASE_URL) {
        throw "API_BASE_URL must be set when RUN_ACCEPTANCE_CHECKS=true."
    }
    $ReadinessArgs = @(
        "scripts\production-readiness-check.py",
        "--base-url",
        $env:API_BASE_URL,
        "--require-approved-model",
        "--require-approved-artifact",
        "--require-prediction-run",
        "--require-monitoring",
        "--require-admin-governance"
    )
    if ($env:ALLOW_STALE_DATA -eq "true") { $ReadinessArgs += "--allow-stale" }
    if ($env:REQUIRE_POLICY_LINKS -eq "true") { $ReadinessArgs += "--require-policy-links" }
    if ($env:REQUIRE_ENRICHED_DATA -ne "false") { $ReadinessArgs += "--require-enriched-data" }
    if ($env:REQUIRE_NO_CRITICAL_ALERTS -ne "false") { $ReadinessArgs += "--require-no-critical-alerts" }
    & $Python @ReadinessArgs
}

if ($env:RELEASE_RECORD_PATH) {
    if (!$env:API_BASE_URL) {
        throw "API_BASE_URL must be set when RELEASE_RECORD_PATH is set."
    }
    & $Python scripts\release-record.py --base-url $env:API_BASE_URL --output $env:RELEASE_RECORD_PATH
}

Write-Host "Staging release checks completed."
