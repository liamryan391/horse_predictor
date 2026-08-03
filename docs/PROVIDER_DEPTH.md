# Provider Depth And Data Enrichment

Phase 13 adds a provider-depth layer around the existing sample-compatible race schema. The goal is to improve model inputs and operator checks without committing to unlicensed redistribution of provider data.

## Current Enrichment Fields

`race_enrichment.py` adds deterministic local enrichment for:

- `country`
- `course_latitude`
- `course_longitude`
- `distance_bucket`
- `going_category`
- `race_type`

The model uses `country`, `distance_bucket`, `going_category`, and `race_type` as categorical inputs. Latitude and longitude are exposed for weather/provider checks and data-quality visibility.

Known courses currently include Aintree, Ascot, Cheltenham, Doncaster, Epsom, Goodwood, Kempton, Lingfield, Newbury, Newmarket, York, Curragh, and Leopardstown.

## Weather Provider Spike

The provider-depth check can query the Open-Meteo historical archive API for daily weather fields by course coordinate and race date. The current check uses daily weather code, mean temperature, precipitation sum, max wind speed, and shallow soil moisture.

```powershell
.\.venv\Scripts\python.exe scripts\provider-depth-check.py --track York --race-date 2025-03-18
```

For local checks without internet access:

```powershell
.\.venv\Scripts\python.exe scripts\provider-depth-check.py --track York --race-date 2025-03-18 --skip-weather
```

Source reference: https://open-meteo.com/en/docs/historical-weather-api

## Data Quality API

`GET /api/v1/data-quality` reports:

- row-quality issues for historical and current race tables
- enrichment coverage by table
- latest provider freshness by provider/table from `api_ingestion_runs`

The React Evaluation view shows the same enrichment coverage and provider freshness so operators can see when model inputs are thin before trusting rankings.

## Production Gate

Use the enrichment gate once production has enough provider coverage:

```powershell
.\.venv\Scripts\python.exe scripts\production-readiness-check.py `
  --base-url https://horse-predictor-api.example.com `
  --require-approved-model `
  --require-approved-artifact `
  --require-prediction-run `
  --require-enriched-data
```

The default threshold is 75% coverage for `country`, `distance_bucket`, `going_category`, and `race_type`. Override it with `--min-enrichment-coverage`.

## Current Limits

- Racing provider credentials are still required for real The Racing API or OurHub ingestion.
- Official ratings imports remain gated by licensing and source format decisions.
- Odds movement snapshots remain future work until provider terms permit storage and display.
- Course metadata is a curated starter list and should be expanded as provider coverage grows.
