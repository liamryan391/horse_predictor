# Backend API

Phase 5 introduces a versioned API contract while keeping the original `/api/...` paths available for compatibility. New clients should use `/api/v1/...`.

FastAPI serves interactive OpenAPI documentation at:

```text
http://127.0.0.1:8000/docs
```

## Versioned Endpoints

Core endpoints:

- `GET /api/v1/health`
- `GET /api/v1/ready`
- `GET /api/v1/summary`
- `GET /api/v1/data-quality`
- `GET /api/v1/safeguards`
- `GET /api/v1/meetings`
- `GET /api/v1/races`
- `GET /api/v1/race-card`
- `GET /api/v1/predictions`
- `GET /api/v1/prediction-runs`
- `GET /api/v1/prediction-runs/{prediction_run_id}`
- `GET /api/v1/bet-journal`
- `GET /api/v1/entities/{entity_type}/{name}`
- `GET /api/v1/model`
- `GET /api/v1/model/evaluation`
- `GET /api/v1/model/registry`
- `GET /api/v1/trends`
- `GET /api/v1/ingestion-status`

Write endpoints:

- `POST /api/v1/bet-journal`
- `PATCH /api/v1/bet-journal/{bet_id}`
- `DELETE /api/v1/bet-journal/{bet_id}`

Administrative endpoints:

- `POST /api/v1/admin/seed-sample`
- `POST /api/v1/admin/model/evaluation`
- `POST /api/v1/admin/model/{model_version_id}/approve`
- `POST /api/v1/admin/prediction-runs`

The same endpoints are also mounted under `/api/...` for current frontend compatibility.

## Filtering And Pagination

List-style endpoints return a `page` object with:

- `limit`
- `offset`
- `returned`
- `total`

Common query parameters:

```text
limit=100
offset=0
track=York
race_date=2026-05-05
horse=Golden
sort_by=suggested_rank
direction=asc
```

Examples:

```powershell
Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8000/api/v1/predictions?track=York&limit=10"
Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8000/api/v1/entities/jockey/A.%20Lee"
```

## Request IDs And Errors

Every response includes an `X-Request-ID` header. Clients can also send their own `X-Request-ID` header.

Error responses use one envelope:

```json
{
  "error": {
    "requestId": "request-id",
    "statusCode": 422,
    "detail": "Error detail"
  }
}
```

## Security Controls

The API avoids returning raw database URLs or credentials. CORS origins are configured with `BACKEND_CORS_ORIGINS`, and deployed API hosts are configured with `ALLOWED_HOSTS`.

Administrative endpoints use bearer-token auth when `API_AUTH_TOKEN` is set:

```powershell
Invoke-WebRequest -Method POST -Headers @{ Authorization = "Bearer $env:API_AUTH_TOKEN" } -UseBasicParsing "http://127.0.0.1:8000/api/v1/admin/seed-sample"
```

For staging or production, `APP_ENV=staging` or `APP_ENV=production` requires:

- managed SQL `DATABASE_URL`
- HTTPS `BACKEND_CORS_ORIGINS`
- explicit `ALLOWED_HOSTS`
- non-placeholder `API_AUTH_TOKEN` with at least 32 characters

The API also rejects oversized requests through `MAX_REQUEST_BODY_BYTES`, sanitizes incoming `X-Request-ID` values, applies baseline security headers, and uses constant-time comparison for administrative bearer tokens.

## Rate Limiting

`API_RATE_LIMIT_PER_MINUTE` controls a lightweight in-process per-client rate limit. The local default is `240`.

Use a gateway or hosting-platform limiter for production traffic; the built-in limiter is a development and single-process safety layer.

## Product Safeguards

`GET /api/v1/safeguards` returns the responsible-use notice, model limitations, data licensing notice, privacy notice, terms notice, and optional policy links configured by:

- `RESPONSIBLE_GAMBLING_URL`
- `PRIVACY_POLICY_URL`
- `TERMS_OF_USE_URL`
- `DATA_LICENSE_REFERENCE`

The React Responsible Use page reads this endpoint and falls back to the built-in launch-safe wording if the API is unavailable.

## Data Quality

`GET /api/v1/data-quality` returns row-quality issues, enrichment coverage, and provider freshness:

- `tables[].issues` contains required-field, invalid odds/result, duplicate identity, and enrichment-coverage issues.
- `tables[].coverage` reports coverage for course country, coordinates, distance bucket, going category, and race type.
- `providerFreshness` reports the latest ingestion status per provider/table from `api_ingestion_runs`.

The React Evaluation view displays this response beside model registry and prediction-run snapshots.

## Model Registry

`GET /api/v1/model` reports the current serving mode. In local development it can fall back to `in_memory`; in staging and production it should report `artifact` after an approved model artifact is available.

`GET /api/v1/model/registry` returns persisted model evaluation snapshots and artifact metadata. Use `POST /api/v1/admin/model/evaluation` to record the current candidate metrics and write a model artifact, then `POST /api/v1/admin/model/{model_version_id}/approve` to approve a reviewed version after artifact integrity checks pass. See [MODEL_OPERATIONS.md](MODEL_OPERATIONS.md) for the release workflow.

## Prediction Runs

`GET /api/v1/prediction-runs` lists persisted scoring snapshots, and `GET /api/v1/prediction-runs/{prediction_run_id}` returns the runner-level rows for one run.

Use `POST /api/v1/admin/prediction-runs` to record the current scored race card after data import or model approval. Add `require_approved_model=true` when staging or production should reject snapshots that were not scored with the approved model artifact. See [PREDICTION_OPERATIONS.md](PREDICTION_OPERATIONS.md) for the run workflow.

## Bet Journal

`GET /api/v1/bet-journal` lists server-side journal rows from `user_bets`. The local app currently uses the operator-local account key, and future authentication should replace that with user-owned accounts before personal bet history is stored.

Create, settle, and remove rows with:

```powershell
Invoke-WebRequest -Method POST -ContentType "application/json" -Body '{"horse":"Golden Arrow","track":"York","raceDate":"2026-08-03","stake":10,"odds":3.5,"status":"open"}' -UseBasicParsing "http://127.0.0.1:8000/api/v1/bet-journal"
Invoke-WebRequest -Method PATCH -ContentType "application/json" -Body '{"status":"won","closingOdds":3.1}' -UseBasicParsing "http://127.0.0.1:8000/api/v1/bet-journal/1"
Invoke-WebRequest -Method DELETE -UseBasicParsing "http://127.0.0.1:8000/api/v1/bet-journal/1"
```

Rows can store manual runner context, stake, placed odds, closing odds, settlement status, notes, and optional links to prediction runs, prediction-run entries, model versions, or normalized race entries.
