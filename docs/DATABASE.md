# Database Platform

Phase 2 introduces an Alembic-managed database platform while keeping the current flat race tables available for the existing API, frontend, and Streamlit diagnostic dashboard.

## Compatibility Tables

The app still reads from:

- `races_historical`
- `races_current`
- `ingestion_log`

These tables remain intentionally flat so the current model and UI keep working during the migration to a normalized model.

## Normalized Tables

The initial normalized schema adds:

- `courses`
- `race_meetings`
- `races`
- `race_entries`
- `horses`
- `jockeys`
- `trainers`
- `owners`
- `historical_results`
- `odds_snapshots`
- `user_bets`
- `api_ingestion_runs`
- `model_versions`
- `prediction_runs`
- `model_evaluation_results`

The normalized tables give future phases a place to store provider identifiers, race-card entities, odds snapshots, model versions, prediction runs, evaluation metrics, and user bet history without overloading the flat model tables.

## Migrations

Run migrations after installing dependencies:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
```

Linux/macOS:

```bash
.venv/bin/python -m alembic upgrade head
```

The first migration is intentionally safe against an existing local SQLite database. It creates missing tables with `checkfirst=True` and records the Alembic version.

## Ingestion Writes

Sample ingestion still replaces the local compatibility tables so demos are repeatable.

Provider ingestion now defaults to upserting rows by:

- `race_date`
- `track`
- `distance`
- `horse`
- `source`

This avoids wiping historical/current rows every time an API provider refresh runs. Later provider adapters should replace this natural key with stable provider race and runner identifiers as soon as the API contract supplies them.
