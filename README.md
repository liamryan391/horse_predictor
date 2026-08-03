# Horse Predictor

Professional horse racing intelligence platform powered by API ingestion, MySQL-ready storage, a FastAPI backend, and a React/TypeScript dashboard.

## Current architecture

- `data_pipeline.py`: pulls sample or API data and writes SQL tables.
- `racing_storage.py`: SQLAlchemy storage layer for MySQL in production and SQLite for local development.
- `api.py`: FastAPI service for predictions, prediction-run history, bet journal records, trends, race cards, data quality, monitoring, admin governance, and ingestion health.
- `monitoring.py`: drift checks and in-process API metrics used by the monitoring endpoint.
- `frontend/`: React + TypeScript racing workspace with rankings, race centre, race cards, evaluation, monitoring, provider freshness, bet journal, and admin console pages.
- `horse_racing_app.py`: legacy Streamlit dashboard kept for quick internal checks.

## Database choice

MySQL is the chosen production database. It is widely supported, easy to host, and works cleanly with Python through SQLAlchemy and PyMySQL.

For local development, the project still defaults to `horse_racing.db` so you can run everything before installing MySQL. To use MySQL, set:

```powershell
$env:DATABASE_URL="mysql+pymysql://horse_user:replace_this_password@localhost:3306/horse_predictor"
```

## Fixing the pandas error

Run project commands through the virtual environment:

```powershell
.\.venv\Scripts\python.exe data_pipeline.py --provider sample
```

Do not run this project with:

```powershell
c:\python313\python.exe
```

That system Python does not have the repo dependencies installed. See [DEVNOTES.md](DEVNOTES.md) for the VS Code setup.

## Setup

```powershell
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For a one-command Windows startup:

```powershell
.\scripts\start-dev.ps1
```

Seed the local database:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe data_pipeline.py --provider sample
```

Model artifacts are written to `model_artifacts/` by default when an admin records a model evaluation snapshot. The directory is ignored by git.

Install frontend dependencies:

```powershell
cd frontend
npm.cmd install
```

## Run the app

Backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd frontend
npm.cmd run dev
```

Open:

```text
http://127.0.0.1:5173
```

Docker Compose is also available for running FastAPI, React, and MySQL together:

```powershell
docker compose up --build
```

## API ingestion

OurHub Racing race-card ingestion:

```powershell
$env:HORSE_API_KEY="your_api_key"
.\.venv\Scripts\python.exe data_pipeline.py --provider ourhub --days-ahead 0
```

The Racing API:

```powershell
$env:RACING_API_USERNAME="your_username"
$env:RACING_API_PASSWORD="your_password"
.\.venv\Scripts\python.exe data_pipeline.py --provider theracingapi
```

Run continuously with an hourly refresh:

```powershell
.\.venv\Scripts\python.exe data_pipeline.py --provider theracingapi --repeat-hourly
```

Remote providers support retry, timeout, rate-limit, and pagination controls:

```powershell
.\.venv\Scripts\python.exe data_pipeline.py --provider theracingapi --retry-attempts 5 --min-request-interval-seconds 1
```

## Required data fields

Historical rows need:

- `race_date`
- `track`
- `distance`
- `surface`
- `horse`
- `jockey`
- `owner`
- `trainer`
- `odds`
- `finishing_position`

Current race-card rows need the same fields except `finishing_position`.

Optional model features:

- `horse_age`
- `horse_weight`
- `draw`
- `speed_rating`
- `class_rating`
- `days_since_last_run`
- `past_bets_count`
- `past_bets_profit`
- `weather`

Phase 13 also derives model enrichment fields from the above data: course country/coordinates, distance bucket, going category, and race type.

## Notes

This is decision-support software, not guaranteed betting advice. Model quality depends on the amount, accuracy, and freshness of historical results. For production, replace shared bearer tokens with full user authentication, managed hosting, larger provider history, and governed model promotion.

See [ROADMAP.md](ROADMAP.md) for the planned path from prototype to production-ready platform.
See [ROADMAP02.md](ROADMAP02.md) for the next build roadmap after Phase 11 and [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md) for longer-term development ideas.
See [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) for imported project context and [docs/WORKFLOW.md](docs/WORKFLOW.md) for the branch and PR workflow.
See [docs/SETUP.md](docs/SETUP.md) for Windows, Linux/macOS, Docker, and configuration setup.
See [docs/DATABASE.md](docs/DATABASE.md) for schema, migration, and ingestion-write notes.
See [docs/INGESTION.md](docs/INGESTION.md) for provider adapter, validation, retry, and failure logging notes.
See [docs/PROVIDER_DEPTH.md](docs/PROVIDER_DEPTH.md) for course metadata, weather checks, enrichment coverage, and provider-depth gates.
See [docs/MODELING.md](docs/MODELING.md) for leakage checks, holdout evaluation, and backtesting metric notes.
See [docs/API.md](docs/API.md) for versioned backend endpoints, pagination, request IDs, and API security notes.
See [docs/FRONTEND.md](docs/FRONTEND.md) for the professional workspace, bet journal, theme, and frontend UX notes.
See [docs/MONITORING.md](docs/MONITORING.md) for drift reports, operator alerts, API metrics, and OpenTelemetry configuration.
See [docs/ADMIN_GOVERNANCE.md](docs/ADMIN_GOVERNANCE.md) for the admin console, bearer-token roles, audit log, and governance checks.
See [docs/CI_RELEASE.md](docs/CI_RELEASE.md) for GitHub Actions CI, staging acceptance, release records, and visual smoke gates.
See [docs/TESTING.md](docs/TESTING.md) for automated backend, data/model, frontend utility, and browser smoke checks.
See [docs/CLOUD_STAGING.md](docs/CLOUD_STAGING.md) for staging deployment, migration, ingestion worker, and observability notes.
See [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md) for security, safeguards, launch acceptance, and rollback gates.
See [docs/MODEL_OPERATIONS.md](docs/MODEL_OPERATIONS.md) for persisted model artifacts, approval, and registry workflow.
See [docs/PREDICTION_OPERATIONS.md](docs/PREDICTION_OPERATIONS.md) for persisted prediction-run snapshots and scoring audit workflow.
