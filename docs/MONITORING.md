# Monitoring And Drift

Phase 16 adds an operator snapshot for API health, data freshness, provider runs, model readiness, and drift.

## Endpoint

Use:

```text
GET /api/v1/monitoring
```

The response includes:

- `status`: worst current signal across drift and alerts.
- `metrics`: dashboard-ready operator metrics.
- `alerts`: warning or critical signals for drift, stale data, provider failures, missing governance evidence, model-serving blocks, API error rate, and slow requests.
- `drift.featureDrift`: reference-vs-current checks for odds, implied probability, field size, ratings, draw, age, weight, course context, going, race type, and weather.
- `drift.predictionDrift`: reference-vs-current checks for win probability, model odds, and value edge when model serving is available.
- `apiMetrics`: in-process request counts, status buckets, top paths, average latency, and recent requests.

The React workspace has a Monitoring tab that reads this endpoint.

## Configuration

Local defaults are intentionally permissive:

```text
MONITORING_DRIFT_WARNING_THRESHOLD=0.35
MONITORING_DRIFT_CRITICAL_THRESHOLD=0.75
MONITORING_SLOW_REQUEST_MS=1000
MONITORING_MAX_ERROR_RATE=0.05
OTEL_ENABLED=false
OTEL_SERVICE_NAME=horse-predictor-api
```

Set `OTEL_ENABLED=true` only when OpenTelemetry FastAPI instrumentation is installed and the runtime has a collector/export path configured. If the package is unavailable, the API logs `otel_unavailable` and keeps serving.

## Readiness Checks

Run the standard smoke:

```powershell
.\.venv\Scripts\python.exe scripts\production-readiness-check.py --base-url http://127.0.0.1:8000
```

Add `--require-monitoring` when staging or production must prove the monitoring contract is populated. Add `--require-no-critical-alerts` when release acceptance should fail on critical or blocked monitoring alerts.

## Drift Method

Numeric drift is scored from normalized mean shift, median shift, and missing-rate delta. Categorical drift is scored from the largest distribution share change plus categories that appear in current data but not in the historical baseline. This is a lightweight in-repo gate, not a replacement for a full model-monitoring platform.
