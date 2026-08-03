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
- `GET /api/v1/monitoring`
- `GET /api/v1/safeguards`
- `GET /api/v1/auth/session`
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
- `GET /api/v1/broker/status`
- `GET /api/v1/broker/raw-payloads`
- `GET /api/v1/broker/raw-payload-shape`
- `GET /api/v1/normalized/status`
- `GET /api/v1/normalized/race-entries`

Write endpoints:

- `POST /api/v1/bet-journal`
- `PATCH /api/v1/bet-journal/{bet_id}`
- `DELETE /api/v1/bet-journal/{bet_id}`

Administrative endpoints:

- `GET /api/v1/admin/session`
- `GET /api/v1/admin/accounts`
- `GET /api/v1/admin/governance`
- `GET /api/v1/admin/audit-log`
- `POST /api/v1/admin/accounts`
- `POST /api/v1/admin/seed-sample`
- `POST /api/v1/admin/model/evaluation`
- `POST /api/v1/admin/model/{model_version_id}/approve`
- `POST /api/v1/admin/model/{model_version_id}/supersede`
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

Account bearer tokens are the preferred auth path. `GET /api/v1/auth/session` returns the resolved actor, account key, account id, roles, auth mode, and environment. Account records are stored in `operator_accounts` with hashed tokens only.

Create or update a local operator account with:

```powershell
.\.venv\Scripts\python.exe scripts\account-smoke-check.py --database-url horse_racing.db --account-key liam --display-name "Liam" --roles admin,journal --token "replace-with-a-long-local-token" --privacy-acknowledged
```

Use that token on admin or journal requests:

```powershell
Invoke-WebRequest -Method GET -Headers @{ Authorization = "Bearer $env:ACCOUNT_AUTH_TOKEN"; "X-Account-Actor" = "Liam" } -UseBasicParsing "http://127.0.0.1:8000/api/v1/auth/session"
```

Roles are `viewer`, `journal`, `operator`, `admin`, and `release-approver`. Admin implies all operator capabilities; journal can read/write only its account-owned bet journal rows.

`API_AUTH_TOKEN` and `JOURNAL_AUTH_TOKEN` remain supported as legacy local-development bearer tokens. In local development only, empty admin and journal tokens fall back to a `local-dev` operator context so the app remains runnable before secrets exist.

For staging or production, `APP_ENV=staging` or `APP_ENV=production` requires:

- managed SQL `DATABASE_URL`
- HTTPS `BACKEND_CORS_ORIGINS`
- explicit `ALLOWED_HOSTS`
- `ACCOUNT_AUTH_ENABLED=true` for named account tokens
- non-empty `JOURNAL_ACCOUNT_KEY`

The API also rejects oversized requests through `MAX_REQUEST_BODY_BYTES`, sanitizes incoming `X-Request-ID` values, applies baseline security headers, hashes account tokens with SHA-256 before storage, and uses constant-time comparison for legacy local bearer tokens.

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

## Local Broker

`GET /api/v1/broker/status` returns the configured raw-payload cache directory, cached payload count, provider/resource counts, latest cached timestamp, and redacted local AI status.

`GET /api/v1/broker/raw-payloads` lists cached payload envelopes with provider, resource, endpoint, source URL, payload hash, row count, and licensing note.

`GET /api/v1/broker/raw-payload-shape?path=<payload-path>` validates a cached payload hash and returns its structural shape. Add `ai_review=true` only after local AI is configured; the response keeps AI review output separate from provider-supplied facts.

See [LOCAL_DATA_BROKER.md](LOCAL_DATA_BROKER.md) for raw-cache commands and local AI setup.

## Normalized Provider Entities

`GET /api/v1/normalized/status` returns row counts for the normalized provider entity tables, including courses, meetings, races, race entries, historical results, odds snapshots, and named participants.

`GET /api/v1/normalized/race-entries` returns the normalized race-entry read model with synthetic or provider-supplied entity IDs, runner context, latest odds snapshot, and historical result fields when available.

Supported filters:

```text
provider=sample
track=York
race_date=2026-08-03
limit=100
offset=0
```

The existing `/race-card` and `/predictions` endpoints still read the compatibility tables. The normalized endpoints are for audit, migration checks, and future provider-depth work.

## Monitoring And Drift

`GET /api/v1/monitoring` returns dashboard-ready operator metrics, drift checks, and alerts:

- `metrics` summarizes freshness, request behavior, drift-check volume, and alert count.
- `alerts` reports warning and critical signals for drift, stale data, provider failures, missing governance evidence, model-serving blocks, API errors, and slow requests.
- `drift.featureDrift` compares current race-card features against the historical reference set.
- `drift.predictionDrift` compares current prediction distributions against reference predictions when serving is available.
- `apiMetrics` exposes in-process request counts, status buckets, top paths, average latency, and recent request samples.

See [MONITORING.md](MONITORING.md) for thresholds and readiness gates.

## Model Registry

`GET /api/v1/model` reports the current serving mode. In local development it can fall back to `in_memory`; in staging and production it should report `artifact` after an approved model artifact is available.

`GET /api/v1/model/registry` returns persisted model evaluation snapshots and artifact metadata. Use `POST /api/v1/admin/model/evaluation` to record the current candidate metrics and write a model artifact, then `POST /api/v1/admin/model/{model_version_id}/approve` to approve a reviewed version after artifact integrity checks pass. See [MODEL_OPERATIONS.md](MODEL_OPERATIONS.md) for the release workflow.

Use `POST /api/v1/admin/model/{model_version_id}/supersede` to remove a model version from active consideration without deleting its audit history.

## Prediction Runs

`GET /api/v1/prediction-runs` lists persisted scoring snapshots, and `GET /api/v1/prediction-runs/{prediction_run_id}` returns the runner-level rows for one run.

Use `POST /api/v1/admin/prediction-runs` to record the current scored race card after data import or model approval. Add `require_approved_model=true` when staging or production should reject snapshots that were not scored with the approved model artifact. See [PREDICTION_OPERATIONS.md](PREDICTION_OPERATIONS.md) for the run workflow.

## Bet Journal

`GET /api/v1/bet-journal` lists server-side journal rows from `user_bets`. The access context supplies the account key: account tokens use `operator_accounts.account_key`, local development uses `local`, legacy admin tokens use `admin`, and legacy journal tokens use `JOURNAL_ACCOUNT_KEY`.

Create, settle, and remove rows with:

```powershell
Invoke-WebRequest -Method POST -ContentType "application/json" -Body '{"horse":"Golden Arrow","track":"York","raceDate":"2026-08-03","stake":10,"odds":3.5,"status":"open"}' -UseBasicParsing "http://127.0.0.1:8000/api/v1/bet-journal"
Invoke-WebRequest -Method PATCH -ContentType "application/json" -Body '{"status":"won","closingOdds":3.1}' -UseBasicParsing "http://127.0.0.1:8000/api/v1/bet-journal/1"
Invoke-WebRequest -Method DELETE -UseBasicParsing "http://127.0.0.1:8000/api/v1/bet-journal/1"
```

Rows can store manual runner context, stake, placed odds, closing odds, settlement status, notes, and optional links to prediction runs, prediction-run entries, model versions, or normalized race entries.

## Admin Governance

`GET /api/v1/admin/governance` returns the operator session, readiness, summary, monitoring snapshot, recent ingestion rows, model registry rows, prediction runs, and audit events in one console payload.

Admin write routes append rows to `admin_audit_events` with actor, roles, action, resource, request id, status, detail, payload, and timestamp. See [ADMIN_GOVERNANCE.md](ADMIN_GOVERNANCE.md) for the full Phase 17 workflow.
