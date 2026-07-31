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
- `GET /api/v1/safeguards`
- `GET /api/v1/meetings`
- `GET /api/v1/races`
- `GET /api/v1/race-card`
- `GET /api/v1/predictions`
- `GET /api/v1/entities/{entity_type}/{name}`
- `GET /api/v1/model`
- `GET /api/v1/model/evaluation`
- `GET /api/v1/trends`
- `GET /api/v1/ingestion-status`

Administrative endpoint:

- `POST /api/v1/admin/seed-sample`

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
