# Prediction Operations

Phase 11 adds persisted prediction-run snapshots. Predictions are no longer only transient API responses; an operator can record the scored race card and inspect the stored run later. Phase 17 records admin-triggered prediction snapshots in the governance audit log.

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
- linked approved model version id when scoring used an approved artifact
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

For staging or production, require an approved model artifact before the run is recorded:

```powershell
Invoke-WebRequest `
  -Method POST `
  -Headers @{ Authorization = "Bearer $env:API_AUTH_TOKEN" } `
  -UseBasicParsing `
  "http://127.0.0.1:8000/api/v1/admin/prediction-runs?require_approved_model=true"
```

The React Admin Console runs this approved-model path by default and writes an `admin_audit_events` row for the captured run.

## Launch Gate

Production acceptance can require at least one persisted prediction run:

```powershell
.\.venv\Scripts\python.exe scripts\production-readiness-check.py `
  --base-url https://horse-predictor-api.example.com `
  --require-approved-model `
  --require-approved-artifact `
  --require-prediction-run
```

## Current Limits

- Scheduled scoring jobs remain future work; drift monitoring is available through `/api/v1/monitoring`.
- Prediction-run cleanup or archival policy remains future work once production volume is known.
