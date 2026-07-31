# Cloud Staging

Phase 8 prepares the project for a hosted staging environment. The repository now has production-style container and release scaffolding, but the managed database, HTTPS domain, and provider secrets still need to be created in the chosen cloud platform.

## Required Services

- FastAPI backend container from `Dockerfile.api`
- Static frontend container from `frontend/Dockerfile.staging`
- Managed MySQL database
- Scheduled ingestion worker using the same backend image
- Secret store for database, API, provider, and CORS settings

## Environment

Use `.env.staging.example` as the staging variable checklist. Store real values in the hosting platform, not in git.

Required staging values:

- `APP_ENV=staging`
- `DATABASE_URL`
- `BACKEND_CORS_ORIGINS`
- `ALLOWED_HOSTS`
- `API_AUTH_TOKEN`
- `VITE_API_BASE_URL`
- provider credentials when moving beyond `HORSE_API_PROVIDER=sample`

Recommended staging values:

- `LOG_FORMAT=json`
- `MAX_REQUEST_BODY_BYTES=1048576`
- `DATA_FRESHNESS_MAX_AGE_HOURS=24`
- `API_RATE_LIMIT_PER_MINUTE=240`
- `RESPONSIBLE_GAMBLING_URL`, `PRIVACY_POLICY_URL`, and `TERMS_OF_USE_URL` before any public launch review

## Release Flow

Run migrations before exposing the backend:

```powershell
$env:DATABASE_URL="mysql+pymysql://..."
.\scripts\staging-release.ps1
```

For Linux hosts:

```bash
DATABASE_URL="mysql+pymysql://..." ./scripts/staging-release.sh
```

Set `SEED_SAMPLE_DATA=true` only for an empty demonstration environment.

## Staging Compose Rehearsal

Validate the staging manifest shape with:

```powershell
docker compose --env-file .env.staging.example -f docker-compose.staging.yml config
```

The staging compose file expects a managed `DATABASE_URL`; it does not provision MySQL. Local database rehearsal remains in `docker-compose.yml`.

## Ingestion Worker

The scheduled worker command is:

```powershell
.\scripts\run-ingestion-worker.ps1
```

The worker uses a database-backed `job_locks` table through `data_pipeline.py --lock-name ingestion-worker`, so overlapping scheduled runs skip instead of duplicating provider writes.

## Observability

Staging uses JSON logs when `APP_ENV=staging` or `LOG_FORMAT=json`.

Monitor:

- `/api/v1/health`
- `/api/v1/ready`
- `/api/v1/ingestion-status`
- `/api/v1/safeguards`
- `/api/v1/model/registry`
- `/api/v1/model/evaluation`
- `/api/v1/prediction-runs`
- frontend freshness indicator from `/api/v1/summary`

The summary endpoint returns `dataFreshness.status`, `ageHours`, and `maxAgeHours`. The workspace shows this as the Freshness metric.

## Acceptance Checklist

- Backend deploy succeeds.
- Frontend static build serves `/health`.
- Managed MySQL connection works.
- `alembic upgrade head` reaches the latest revision.
- `/api/v1/health` returns `ok` before model approval.
- `/api/v1/safeguards` returns responsible-use, licensing, privacy, and terms notices.
- A candidate model snapshot and artifact can be recorded through `POST /api/v1/admin/model/evaluation`.
- The reviewed candidate can be approved through `POST /api/v1/admin/model/{model_version_id}/approve` after artifact checksum and feature-schema checks pass.
- `/api/v1/ready` returns `ok` after artifact-backed model approval.
- A prediction snapshot can be recorded through `POST /api/v1/admin/prediction-runs?require_approved_model=true` after model approval.
- `MODEL_ARTIFACT_DIR` should point at durable storage; the staging compose file mounts `/app/model_artifacts` as a named volume.
- `python scripts/production-readiness-check.py --base-url <api-url> --require-approved-artifact` passes.
- Ingestion worker records a successful run.
- Frontend can load meetings, race cards, predictions, model evaluation, and trends.
- HTTPS domain and CORS origins match exactly.
- Future visual gate: make the `agent-browser` CLI available on PATH, then run the browser smoke commands from `docs/TESTING.md`.
