# Prediction Operations

Phase 11 adds persisted prediction-run snapshots. Predictions are no longer only transient API responses; an operator can record the scored race card and inspect the stored run later.

## Prediction Run History

Public list endpoint:

```text
GET /api/v1/prediction-runs
```

Public detail endpoint:

```text
GET /api/v1/prediction-runs/{prediction_run_id}
```

Each run records:

- prediction run id
- linked approved model version id when one exists
- run timestamp
- source
- runner count
- top runner, probability, and value edge

The run detail includes the persisted runner-level scoring rows: race date, track, distance, runner, market odds, model probability, model odds, value edge, and suggested rank.

## Recording A Prediction Snapshot

Use the administrative endpoint after a fresh race-card import or before launch acceptance:

```powershell
Invoke-WebRequest `
  -Method POST `
  -Headers @{ Authorization = "Bearer $env:API_AUTH_TOKEN" } `
  -UseBasicParsing `
  "http://127.0.0.1:8000/api/v1/admin/prediction-runs"
```

Optional filters let an operator snapshot a single track, date, or runner search:

```powershell
Invoke-WebRequest `
  -Method POST `
  -Headers @{ Authorization = "Bearer $env:API_AUTH_TOKEN" } `
  -UseBasicParsing `
  "http://127.0.0.1:8000/api/v1/admin/prediction-runs?track=York"
```

For staging or production, require approved model metadata before the run is recorded:

```powershell
Invoke-WebRequest `
  -Method POST `
  -Headers @{ Authorization = "Bearer $env:API_AUTH_TOKEN" } `
  -UseBasicParsing `
  "http://127.0.0.1:8000/api/v1/admin/prediction-runs?require_approved_model=true"
```

## Launch Gate

Production acceptance can require at least one persisted prediction run:

```powershell
.\.venv\Scripts\python.exe scripts\production-readiness-check.py `
  --base-url https://horse-predictor-api.example.com `
  --require-approved-model `
  --require-prediction-run
```

## Current Limits

- The prediction run links to approved model metadata when available, but the scoring model is still trained in process from the current database.
- Serialized model artifacts remain future work.
- Scheduled scoring jobs and drift monitoring remain future work.
