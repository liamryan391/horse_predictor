# CI, Release, And Visual Gates

Phase 18 turns the manual validation list into repeatable checks.

## Pull Request CI

`.github/workflows/ci.yml` runs on pull requests into `development` or `main`, and on direct pushes to `development` or `main`.

The workflow has three jobs:

- Backend: installs Python dependencies, compiles key modules, runs Alembic migrations, and runs the Python test suite.
- Frontend: installs with `npm ci`, runs frontend utility tests, and builds the Vite app.
- Scripts And Docker Config: checks shell script syntax, validates Docker Compose manifests, and runs `git diff --check`.

Dependabot is already configured for Python, frontend npm, and GitHub Actions updates in `.github/dependabot.yml`.

## Staging Acceptance

`.github/workflows/staging-acceptance.yml` is a manual workflow. It expects:

- a staging API URL as `api_base_url`
- an admin-capable account token stored as `ACCOUNT_AUTH_TOKEN` or legacy `API_AUTH_TOKEN`
- a staging API that already has migrated schema, approved model artifact, prediction-run evidence, monitoring, and admin governance endpoints

The workflow runs `scripts/production-readiness-check.py` with release-grade gates, then writes `release-records/staging-release-record.json` and uploads it as an artifact.

Before running the workflow against a new environment, run:

```powershell
.\.venv\Scripts\python.exe scripts\staging-env-check.py --env-file .env.staging --require-release-token --require-policy-links
.\.venv\Scripts\python.exe scripts\managed-db-rehearsal.py --mode roundtrip --source-url $env:DATABASE_URL --restore-url $env:RESTORE_DATABASE_URL --backup-path backups\staging-rehearsal.sql
```

The second command is a dry-run unless `--execute` is added.

## Local Release Command

The existing staging release scripts can now run acceptance gates and write release evidence.

PowerShell:

```powershell
$env:DATABASE_URL="mysql+pymysql://..."
$env:API_BASE_URL="https://horse-predictor-api-staging.example.com"
$env:ACCOUNT_AUTH_TOKEN="your-admin-account-token"
$env:RUN_ACCEPTANCE_CHECKS="true"
$env:RELEASE_RECORD_PATH="release-records/staging-release-record.json"
.\scripts\staging-release.ps1
```

Bash:

```bash
DATABASE_URL="mysql+pymysql://..." \
API_BASE_URL="https://horse-predictor-api-staging.example.com" \
ACCOUNT_AUTH_TOKEN="your-admin-account-token" \
RUN_ACCEPTANCE_CHECKS=true \
RELEASE_RECORD_PATH="release-records/staging-release-record.json" \
./scripts/staging-release.sh
```

Optional environment switches:

- `ALLOW_STALE_DATA=true`
- `REQUIRE_POLICY_LINKS=true`
- `REQUIRE_ENRICHED_DATA=false`
- `REQUIRE_NO_CRITICAL_ALERTS=false`

## Release Record

`scripts/release-record.py` captures:

- UTC generation time
- Git branch and commit SHA
- Alembic head/current output
- readiness and data freshness
- monitoring status and alert codes
- current serving model and approved model artifact metadata
- latest prediction run
- ingestion rows
- optional admin governance and latest audit action

Run it directly with:

```powershell
.\.venv\Scripts\python.exe scripts\release-record.py --base-url http://127.0.0.1:8000 --admin-token $env:ACCOUNT_AUTH_TOKEN --output release-records/local-release-record.json
```

Release records are ignored by git and should be stored as CI artifacts or release evidence.

## Visual Gate

`scripts/visual-smoke-check.py` uses the local `agent-browser` CLI to verify the frontend loads, has no Vite/Next/Webpack error overlay, can navigate core tabs, can open Race Card and Evaluation workspace views, and still renders at a mobile viewport.

Run it after backend and frontend previews are live:

```powershell
$env:ACCOUNT_AUTH_TOKEN="your-admin-account-token"
.\.venv\Scripts\python.exe scripts\visual-smoke-check.py --base-url http://127.0.0.1:5173 --require-admin --screenshot --verbose
```

The script checks Workspace, Race Card, Evaluation, Bet Journal, Responsible Use, and Admin when a token is provided. It closes the browser automatically unless `--keep-open` is set.
