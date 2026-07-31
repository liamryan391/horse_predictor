# Model Operations

Phase 10 adds an auditable model registry foundation. The API can now persist candidate evaluation snapshots, list model versions with metrics, and mark one version as approved.

## Model Registry

Public registry endpoint:

```text
GET /api/v1/model/registry
```

The response includes:

- model version id
- name
- algorithm
- status: `candidate`, `approved`, or `superseded`
- training date window
- feature count
- persisted evaluation metrics
- created and updated timestamps

The React Evaluation view shows the latest registry rows below the live model evaluation metrics.

## Recording A Candidate Snapshot

Use the administrative endpoint after a fresh data import or before release review:

```powershell
Invoke-WebRequest `
  -Method POST `
  -Headers @{ Authorization = "Bearer $env:API_AUTH_TOKEN" } `
  -UseBasicParsing `
  "http://127.0.0.1:8000/api/v1/admin/model/evaluation"
```

This trains the current model, runs chronological holdout evaluation, stores a `candidate` row in `model_versions`, and stores non-empty metrics in `model_evaluation_results`.

## Approving A Model

Approve a candidate only after reviewing the metrics against the market baseline and checking the data window:

```powershell
Invoke-WebRequest `
  -Method POST `
  -Headers @{ Authorization = "Bearer $env:API_AUTH_TOKEN" } `
  -UseBasicParsing `
  "http://127.0.0.1:8000/api/v1/admin/model/1/approve"
```

Approving one model marks any previous `approved` model as `superseded`.

## Launch Gate

Production acceptance can require an approved model version:

```powershell
.\.venv\Scripts\python.exe scripts\production-readiness-check.py `
  --base-url https://horse-predictor-api.example.com `
  --require-policy-links `
  --require-approved-model `
  --require-hsts
```

## Current Limits

- The registry stores evaluation metadata and metrics, not a serialized model artifact.
- There is no automated model promotion; approval remains an explicit administrative action.
- A future phase should persist trained artifacts, record prediction runs, and compare candidate metrics against the current approved model before allowing promotion.
