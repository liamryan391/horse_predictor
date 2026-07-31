# Racing API Ingestion

Phase 3 moves provider-specific ingestion behind adapter classes. The command-line interface stays the same, but provider mapping, validation, retry behavior, and failure logging are now separated from the runner.

## Providers

Supported providers:

- `sample`: local CSV seed data for demos and development.
- `generic`: bearer/API-key JSON endpoints for `/historical-races` and `/current-races`.
- `theracingapi`: username/password API flow for historical results and race cards.
- `ourhub`: API-key race-card flow for course and runner endpoints.

Provider logic lives in `provider_adapters.py`.

## Reliability Controls

All remote providers support:

- request timeout
- retry attempts
- linear backoff
- minimum request interval for rate-limit friendliness
- bounded pagination for providers that return page hints

Environment variables:

```text
HORSE_API_TIMEOUT_SECONDS=30
HORSE_API_RETRY_ATTEMPTS=3
HORSE_API_RETRY_BACKOFF_SECONDS=1.5
HORSE_API_MIN_REQUEST_INTERVAL_SECONDS=0
HORSE_API_MAX_PAGES=5
```

CLI overrides:

```powershell
.\.venv\Scripts\python.exe data_pipeline.py --provider theracingapi --retry-attempts 5 --min-request-interval-seconds 1
```

## Validation

Provider rows are normalized to the compatibility race schema before database writes. The validator drops rows that are missing core identity fields:

- `race_date`
- `track`
- `horse`
- `finishing_position` for historical rows

The validator records warnings for prediction-useful fields that are missing or invalid:

- `distance`
- `surface`
- `jockey`
- `owner`
- `trainer`
- `odds`

Validation summaries are written into ingestion-run messages when rows are saved.

## Enrichment

Phase 13 adds deterministic enrichment after provider normalization:

- course country and coordinates for known tracks
- distance buckets such as sprint, mile, middle, and staying
- going categories from surface/weather text
- race type buckets for flat turf, all-weather, jumps, and unknown

The enriched columns are used by model feature creation and `/api/v1/data-quality`. The compatibility race tables still store the base race schema; enrichment is derived at read/scoring time so provider licensing and schema decisions can evolve without a database migration for every derived field.

Run the provider-depth smoke check for local course metadata and optional Open-Meteo historical weather:

```powershell
.\.venv\Scripts\python.exe scripts\provider-depth-check.py --track York --race-date 2025-03-18
```

## Failure Logging

Provider fetch failures are recorded in `api_ingestion_runs` with:

- provider name
- `provider_fetch` target
- failure status
- error message
- completion timestamp

This gives the API and dashboard a path to show ingestion health without hiding provider errors.

## Current Limits

The current upsert key is:

- `race_date`
- `track`
- `distance`
- `horse`
- `source`

That is safe enough for current provider smoke tests, but it should be replaced with stable provider race and runner IDs once the chosen provider contract is locked down.
