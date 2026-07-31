# Automated Testing

Phase 7 adds repeatable backend, data/model, provider, storage, migration, and frontend utility checks.

## Backend

Run all Python tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The suite covers:

- API contract shapes and filtering
- prediction-run snapshot persistence and API contracts
- prediction leakage and chronological holdout behavior
- provider fixture flattening and validation summaries
- SQLite repository upsert behavior
- Alembic upgrade/downgrade roundtrip
- sample data quality gates
- model calibration and market-baseline checks
- model artifact save/load determinism and artifact-backed API serving

## Frontend

Run the lightweight frontend utility tests:

```powershell
npm.cmd test --prefix frontend
```

Run the production build:

```powershell
npm.cmd run build --prefix frontend
```

The frontend tests cover browser-local bet journal math, draft validation, and local-storage parsing. UI coverage still relies on Vite build plus browser smoke checks until a dedicated component/e2e runner is introduced.

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

When `agent-browser` is available on PATH, use it for the visual pass:

```powershell
agent-browser open http://127.0.0.1:5173
agent-browser wait --load networkidle
agent-browser snapshot -i
agent-browser close
```

Future check note: `agent-browser` is a required visual smoke gate before staging acceptance, but it is not currently required for backend/frontend automated tests.
