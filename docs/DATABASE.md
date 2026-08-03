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
- `prediction_run_entries`
- `model_evaluation_results`

The normalized tables give future phases a place to store provider identifiers, race-card entities, odds snapshots, model versions, prediction run headers, runner-level prediction entries, evaluation metrics, and user bet history without overloading the flat model tables.

`model_versions` also stores artifact-serving metadata:

- `artifact_uri`
- `artifact_sha256`
- `feature_schema_hash`
- `code_commit_sha`

The database stores artifact references and integrity values; the model artifact files themselves live under `MODEL_ARTIFACT_DIR` or equivalent durable storage.

`user_bets` now backs the server-side Bet Journal. It supports manual journal rows before provider entity matching is complete:

- `account_key` stores the current operator-local ownership scope until real account auth exists.
- `horse`, `track`, and `race_date` capture manual runner context.
- `stake`, `odds_decimal`, `closing_odds_decimal`, `status`, `settled_at`, and `profit_loss` capture settlement state.
- `prediction_run_id`, `prediction_run_entry_id`, `model_version_id`, and nullable `race_entry_id` provide linkage points for model and result audits.
- `notes` and `updated_at` keep operator context without storing personal contact data.

## Migrations

Run migrations after installing dependencies:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
```

Linux/macOS:

```bash
.venv/bin/python -m alembic upgrade head
```

The first migration is intentionally safe against an existing local SQLite database. It creates missing tables with `checkfirst=True` and records the Alembic version. Later migrations add job locks, prediction-run entries, model artifact integrity columns, and server-side Bet Journal fields.

## Ingestion Writes

Sample ingestion still replaces the local compatibility tables so demos are repeatable.

Provider ingestion now defaults to upserting rows by:

- `race_date`
- `track`
- `distance`
- `horse`
- `source`

This avoids wiping historical/current rows every time an API provider refresh runs. Later provider adapters should replace this natural key with stable provider race and runner identifiers as soon as the API contract supplies them.
