# Live Testing Guide

This guide proves that Horse Predictor can move from provider data to database rows, API responses, model scoring, and the frontend workspace.

## Current Authenticated Result

The August 3, 2026 authenticated provider check confirmed:

- OurHub credentials work for live race-card data.
- The redacted provider smoke check passes for OurHub without printing secrets.
- OurHub imported 508 current runners for 7 tracks into a fresh SQLite check database.
- OurHub track labels now normalize to course names instead of full race labels.
- API health, meetings, track filtering, race-card rows, prediction rows, data quality, and provider freshness worked against the live OurHub import.
- The Racing API credentials work for `/v1/courses`, but the current account plan does not allow `/v1/racecards` or `/v1/results`.
- OurHub does not currently provide odds or owner fields through the mapped feed, so value-edge calculations remain unavailable until an odds source is added.

See [LIVE_PROVIDER_CHECK_2026-08-03.md](LIVE_PROVIDER_CHECK_2026-08-03.md) for the exact local check evidence.

## Current Pre-Merge Result

The August 3, 2026 local release check confirmed:

- internet access works from the machine
- Open-Meteo weather enrichment works for known tracks such as York
- supported racing-provider endpoints are reachable
- OurHub and The Racing API correctly reject unauthenticated requests
- no provider credentials are configured in the local environment or `.env`
- sample ingestion writes 6 historical rows and 3 current rows into a fresh SQLite database
- the API reads those rows, finds York as the meeting track, returns the York race card, returns zero rows for a fake track, and produces ranked predictions
- local production-readiness and release-record checks pass against the running app when stale sample data is allowed

The first unauthenticated check is kept here as historical context. The later authenticated check above supersedes the old credential status.

## Provider Credentials

Copy `.env.example` to `.env`, then choose one provider.

OurHub Racing:

```powershell
$env:HORSE_API_PROVIDER="ourhub"
$env:HORSE_API_KEY="your-ourhub-api-key"
```

The adapter uses:

- `GET https://api.ourhub.site/api/course-info/{race_date}`
- `GET https://api.ourhub.site/api/runner-info/{race_date}`

OurHub source reference: https://github.com/TamB10/ourhub-racing-api

The Racing API:

```powershell
$env:HORSE_API_PROVIDER="theracingapi"
$env:RACING_API_USERNAME="your-username"
$env:RACING_API_PASSWORD="your-password"
```

The adapter uses:

- `GET https://api.theracingapi.com/v1/results`
- `GET https://api.theracingapi.com/v1/racecards`

The Racing API source reference: https://api.theracingapi.com/documentation

Generic JSON provider:

```powershell
$env:HORSE_API_PROVIDER="generic"
$env:HORSE_API_BASE_URL="https://your-provider.example.com"
$env:HORSE_API_KEY="your-api-key"
```

The generic adapter expects:

- `GET /historical-races`
- `GET /current-races?days_ahead=<n>`

Built-in providers use their official API hosts by default, so a stray `HORSE_API_BASE_URL` will not redirect `ourhub` or `theracingapi` traffic. Use `HORSE_API_BASE_URL` for `generic` providers only.

## Provider Smoke Certification

Run the redacted smoke command before a live import. It reports endpoint status, row counts, coverage, validation issues, and pass/fail criteria without printing credential values.

OurHub:

```powershell
.\.venv\Scripts\python.exe scripts\provider-smoke-check.py --provider ourhub --days-ahead 0 --timeout-seconds 20 --retry-attempts 1 --min-request-interval-seconds 0.25
```

Expected current result:

- `course_info` and `runner_info` return HTTP 200
- current rows are greater than zero
- track, distance, surface, horse, jockey, and trainer coverage pass
- odds and owner may be missing on the current OurHub feed

The Racing API:

```powershell
.\.venv\Scripts\python.exe scripts\provider-smoke-check.py --provider theracingapi --history-start 2026-08-01 --history-end 2026-08-03 --timeout-seconds 20 --retry-attempts 1 --max-pages 1 --allow-failure
```

Expected current result on the free account:

- `/v1/courses` returns HTTP 200
- `/v1/racecards` reports `Basic Plan required`
- `/v1/results` reports `Standard Plan required`

Use `--allow-base-url-override --base-url <url>` only when intentionally testing a proxy, staging copy, or mock server for a built-in provider.

## Fresh Live-Check Database

Use a throwaway database first so provider tests do not damage the working demo DB.

```powershell
$env:DATABASE_URL="sqlite:///live-provider-check.db"
.\.venv\Scripts\python.exe -m alembic upgrade head
```

Seed sample data to prove the storage and API path:

```powershell
.\.venv\Scripts\python.exe data_pipeline.py --provider sample --database-url live-provider-check.db --no-csv --disable-lock
```

Then run the selected live provider.

OurHub current race card:

```powershell
.\.venv\Scripts\python.exe data_pipeline.py --provider ourhub --database-url live-provider-check.db --days-ahead 0 --no-csv --disable-lock
```

The Racing API history plus current race card:

```powershell
.\.venv\Scripts\python.exe data_pipeline.py --provider theracingapi --database-url live-provider-check.db --history-start 2026-01-01 --history-end 2026-08-03 --no-csv --disable-lock
```

## Verify Database Writes

```powershell
.\.venv\Scripts\python.exe -c "from racing_storage import table_counts, ingestion_status; import json; db='live-provider-check.db'; print(json.dumps({'counts': table_counts(db), 'ingestion': ingestion_status(db).to_dict(orient='records')}, default=str, indent=2))"
```

Expected result:

- `counts.current` is greater than zero after a race-card import
- `counts.historical` is greater than zero after sample or The Racing API historical import
- `api_ingestion_runs` includes success rows for provider writes
- provider failures are visible with a useful message

## Verify API Track Matching

Start the backend against the live-check database:

```powershell
$env:DATABASE_URL="sqlite:///live-provider-check.db"
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Check the API:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/summary
Invoke-RestMethod http://127.0.0.1:8000/api/v1/meetings
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/race-card?track=York&limit=50"
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/predictions?track=York&limit=50"
Invoke-RestMethod http://127.0.0.1:8000/api/v1/data-quality
Invoke-RestMethod http://127.0.0.1:8000/api/v1/ingestion-status
```

For a real provider run, replace `York` with a track returned by `/api/v1/meetings`.

Pass criteria:

- `/api/v1/meetings` lists the imported tracks
- `/api/v1/race-card?track=<track>` returns only that track
- `/api/v1/race-card?track=NotARealTrack` returns zero rows
- `/api/v1/predictions?track=<track>` returns ranked rows
- `/api/v1/data-quality` reports enrichment coverage
- `/api/v1/ingestion-status` shows the provider status

## Verify The Frontend

Start the frontend:

```powershell
npm.cmd run dev --prefix frontend
```

Open `http://127.0.0.1:5173`, then check:

- Workspace shows historical/current counts
- Race Centre groups runners by race
- Race Card shows imported track rows
- Evaluation shows model metrics and data quality
- Monitoring shows freshness, alerts, drift, and provider status
- Admin loads with a valid admin-capable `ACCOUNT_AUTH_TOKEN`

When `agent-browser` is healthy:

```powershell
.\.venv\Scripts\python.exe scripts\visual-smoke-check.py --base-url http://127.0.0.1:5173 --require-admin --admin-token $env:ACCOUNT_AUTH_TOKEN --verbose
```

If `agent-browser` cannot open the page and reports a daemon connection timeout, run:

```powershell
agent-browser close --all
agent-browser doctor --fix
```

If the daemon still hangs, restart the terminal or Codex Desktop, then rerun the visual gate. Continue API and build checks while treating browser automation as tool-side blocked.

## Release Rehearsal

After a live provider import, run:

```powershell
.\.venv\Scripts\python.exe scripts\production-readiness-check.py --base-url http://127.0.0.1:8000 --require-approved-model --require-approved-artifact --require-prediction-run --require-monitoring --require-admin-governance --admin-token $env:ACCOUNT_AUTH_TOKEN
.\.venv\Scripts\python.exe scripts\release-record.py --base-url http://127.0.0.1:8000 --admin-token $env:ACCOUNT_AUTH_TOKEN --output release-records/local-live-release-record.json
```

Add `--allow-stale` only for local sample data or a rehearsal where freshness is intentionally not release-blocking.

## Known Live-Provider Gaps

- The app currently supports OurHub and The Racing API, and local credentials are present in `.env`.
- `scripts/provider-smoke-check.py` is the standard no-secret provider gate before live imports.
- OurHub currently imports current race-card rows only; historical depth still needs sample data, The Racing API Standard Plan, or another results provider.
- OurHub does not currently provide mapped odds or owner fields, so market-implied probability and value edge are unavailable for OurHub-only cards.
- The Racing API current credentials can access `/v1/courses`, but racecards require Basic Plan and results require Standard Plan.
- The flat compatibility upsert key still serves current API/model reads; Phase 21 mirrors rows into normalized tables with synthetic provider IDs until official provider race and runner IDs are available.
- The frontend track filter depends on normalized provider course names; provider aliases should be mapped during Roadmap03.
- Browser automation is useful, but the local `agent-browser` daemon can still hang and should not be the only release signal until it is stable.
