# Automated Testing

Phase 7 adds repeatable backend, data/model, provider, storage, migration, and frontend utility checks.

## Backend

Run all Python tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The suite covers:

- API contract shapes and filtering
- server-side bet journal CRUD contracts
- account-auth session, operator-account storage, admin session, governance snapshot, audit-log, and model supersede contracts
- prediction-run snapshot persistence and API contracts
- prediction leakage and chronological holdout behavior
- provider fixture flattening and validation summaries
- SQLite repository upsert behavior and bet journal persistence
- admin audit persistence and audit-table migrations
- operator-account token hashing, role normalization, and account-table migrations
- Alembic upgrade/downgrade roundtrip
- sample data quality gates
- enrichment coverage and course/weather URL checks
- model calibration and market-baseline checks
- model artifact save/load determinism and artifact-backed API serving
- monitoring drift helpers, API request counters, and the `/api/v1/monitoring` contract
- local broker cache helpers, manual JSON normalization, and local AI JSON validation
- normalized provider entity sync, table counts, and normalized race-entry read model
- managed DB rehearsal planning, restore-target safety, staging env validation, and secret redaction

## Frontend

Run the lightweight frontend utility tests:

```powershell
npm.cmd test --prefix frontend
```

Run the production build:

```powershell
npm.cmd run build --prefix frontend
```

The frontend tests cover bet journal math, server payload validation, local fallback parsing, race-centre grouping, race-centre runner sorting, and transparent signal labels. UI coverage still relies on Vite build plus browser smoke checks until a dedicated component/e2e runner is introduced.

## Browser Smoke

Start the API and frontend, then verify:

```powershell
Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:8000/api/v1/health
Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:5173
```

Run the production-readiness API smoke check against a live API:

```powershell
.\.venv\Scripts\python.exe scripts\production-readiness-check.py --base-url http://127.0.0.1:8000
```

Add `--require-approved-model` for staging or production once a model version has been recorded and approved.
Add `--require-approved-artifact` when acceptance should also prove that the approved model has artifact integrity metadata.
Add `--require-prediction-run` when acceptance should also prove that at least one scored race-card snapshot has been persisted.
Add `--require-enriched-data` when acceptance should also prove that core enrichment fields meet the configured coverage threshold.
Add `--require-monitoring` when acceptance should also prove that monitoring metrics and drift checks are populated.
Add `--require-no-critical-alerts` when acceptance should fail on critical or blocked monitoring alerts.
Add `--require-admin-governance` when acceptance should also prove that admin session, governance, and audit-log endpoints work with an admin-capable account token.

Run the provider-depth check with live weather-provider access:

```powershell
.\.venv\Scripts\python.exe scripts\provider-depth-check.py --track York --race-date 2025-03-18
```

Run the redacted provider smoke check before live racing imports:

```powershell
.\.venv\Scripts\python.exe scripts\provider-smoke-check.py --provider ourhub --days-ahead 0 --timeout-seconds 20 --retry-attempts 1
```

Use the smoke check to confirm endpoint reachability, provider-plan limits, row counts, field coverage, and validation issues without printing API keys or passwords.

Run the no-network local broker smoke check:

```powershell
.\.venv\Scripts\python.exe scripts\broker-smoke-check.py --cache-dir .codex_tmp\broker-smoke
```

Run the local account-auth smoke check:

```powershell
.\.venv\Scripts\python.exe scripts\account-smoke-check.py --database-url .codex_tmp\phase22-accounts.db --account-key liam --display-name "Liam" --roles admin,journal --token "phase22-local-token-12345" --privacy-acknowledged
```

Run the local AI smoke check after starting Ollama, LM Studio, or another OpenAI-compatible local server:

```powershell
.\.venv\Scripts\python.exe scripts\local-ai-smoke-check.py --provider ollama --base-url http://127.0.0.1:11434/v1 --model llama3.1
```

With `AI_PROVIDER=disabled`, the AI smoke check confirms the disabled path without making a network request.

Run the normalized provider entity backfill check:

```powershell
.\.venv\Scripts\python.exe scripts\normalized-backfill-check.py --database-url .codex_tmp\phase21-normalized.db --source sample
```

Run Phase 23 deployment guardrails:

```powershell
.\.venv\Scripts\python.exe scripts\staging-env-check.py --env-file .env.staging.example --allow-placeholders --require-release-token
.\.venv\Scripts\python.exe scripts\managed-db-rehearsal.py --mode roundtrip --source-url mysql+pymysql://horse_user:secret@db.example.net:3306/horse_predictor --restore-url mysql+pymysql://horse_restore:secret@restore.example.net:3306/horse_predictor_restore --backup-path backups\dry-run.sql
```

When `agent-browser` is available on PATH, use it for the visual pass:

```powershell
.\.venv\Scripts\python.exe scripts\visual-smoke-check.py --base-url http://127.0.0.1:5173 --verbose
```

Add `--reset-browser` when a previous browser daemon was left open or the page preview reports `127.0.0.1 refused to connect` after the dev servers have been restarted:

```powershell
.\.venv\Scripts\python.exe scripts\visual-smoke-check.py --base-url http://127.0.0.1:5173 --reset-browser --verbose
```

Use `--require-admin --admin-token $env:ACCOUNT_AUTH_TOKEN` when the Admin Console should be included in the visual gate.

The visual gate checks page content, framework overlays, Workspace, Race Centre, Race-Day Board, Race Card, Evaluation, Bet Journal, Responsible Use, and mobile rendering. Admin is included only when a token is supplied.

If `agent-browser` reports a daemon connection timeout or leaves a stuck helper process, run the reset helper, then restart the terminal or Codex Desktop if needed:

```powershell
.\scripts\agent-browser-reset.ps1
```

```bash
scripts/agent-browser-reset.sh
```

Treat a repeated daemon timeout as a browser-automation tool issue when the HTTP API, frontend dev server, and build checks are otherwise green.

For provider-credential rehearsal, database verification, track matching, and live race-card checks, see [LIVE_TESTING.md](LIVE_TESTING.md).

## CI

GitHub Actions CI now runs backend, frontend, script syntax, Docker Compose config, and diff-hygiene checks on pull requests plus direct pushes to `development` and `main`.

See [CI_RELEASE.md](CI_RELEASE.md) for CI jobs, staging acceptance, release-record output, and visual gate details.
