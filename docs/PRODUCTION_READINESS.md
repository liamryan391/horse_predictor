# Production Readiness

Phase 9 turns the staging foundation into a launch checklist. The repository now includes runtime guardrails and automated smoke checks, but the final production launch still depends on the chosen host, data provider contract, and business/legal signoff.

## Security Review

Runtime controls:

- `APP_ENV=staging` or `APP_ENV=production` fails fast unless `DATABASE_URL` is a managed SQL database.
- `BACKEND_CORS_ORIGINS` must be explicitly set and must use HTTPS outside local development.
- `ALLOWED_HOSTS` must be explicitly set in staging and production.
- `ACCOUNT_AUTH_ENABLED=true` is the preferred staging/production auth mode.
- Named operator accounts store hashed bearer tokens and role assignments in `operator_accounts`.
- `API_AUTH_TOKEN` and `JOURNAL_AUTH_TOKEN` are legacy local-development fallback tokens; if `ACCOUNT_AUTH_ENABLED=false` is deliberately used, both must be non-placeholder secrets with at least 32 characters.
- `JOURNAL_ACCOUNT_KEY` must be non-empty for server-side journal ownership.
- `MAX_REQUEST_BODY_BYTES` rejects oversized requests before route handlers run.
- `REQUIRE_APPROVED_MODEL_ARTIFACT=true` requires an approved, loadable model artifact before prediction serving is ready.
- Monitoring thresholds are configurable with `MONITORING_DRIFT_WARNING_THRESHOLD`, `MONITORING_DRIFT_CRITICAL_THRESHOLD`, `MONITORING_SLOW_REQUEST_MS`, and `MONITORING_MAX_ERROR_RATE`.
- API responses include request IDs and baseline security headers.
- Administrative and journal routes resolve bearer tokens to named accounts before applying role checks.
- Governed admin writes record audit rows in `admin_audit_events`, including operator-account changes.

Operational checks:

- Rotate account bearer tokens, provider credentials, and database passwords before production launch.
- Create at least one admin-capable operator account before exposing staging admin routes.
- Use separate MySQL users for application reads/writes, migrations, backups, and provider import jobs.
- Restrict the application database user to the `horse_predictor` schema.
- Enable platform DDoS/rate-limit protections in front of the built-in single-process limiter.
- Keep real secrets only in the hosting platform or local `.env`; never commit them.

Dependency scanning:

- `.github/workflows/ci.yml` runs backend, frontend, script, Docker Compose, and diff-hygiene checks on pull requests plus direct pushes to `development` and `main`.
- `.github/workflows/staging-acceptance.yml` can run release gates and upload a release-record artifact from a staging API URL.
- `.github/dependabot.yml` checks Python, frontend npm, and GitHub Actions dependencies weekly.
- Review Dependabot security PRs before routine version bumps.
- Run `npm.cmd audit` from `frontend/` before release when network access is available.
- Run a Python vulnerability scanner such as `pip-audit` against `requirements.txt` before release when network access is available.

## Backup And Restore Test

Before launch, prove that production data can be restored into an isolated staging database.

1. Create a backup from the managed MySQL production candidate.
2. Restore it into a separate staging or restore-test database.
3. Point `DATABASE_URL` at the restored database.
4. Run migrations:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
```

5. Run acceptance checks:

```powershell
.\.venv\Scripts\python.exe scripts\production-readiness-check.py --base-url https://horse-predictor-api-staging.example.com --require-policy-links --require-approved-artifact --require-enriched-data --require-monitoring --require-admin-governance
```

6. Confirm `/api/v1/summary` reports fresh data, `/api/v1/data-quality` reports acceptable enrichment coverage, `/api/v1/monitoring` reports populated drift checks, `/api/v1/model/evaluation` is acceptable for launch, and `/api/v1/model` reports `servingMode=artifact`.

Phase 23 adds helper checks:

```powershell
.\.venv\Scripts\python.exe scripts\staging-env-check.py --env-file .env.production --require-release-token --require-policy-links --require-provider-credentials
.\.venv\Scripts\python.exe scripts\managed-db-rehearsal.py --mode roundtrip --source-url $env:DATABASE_URL --restore-url $env:RESTORE_DATABASE_URL --backup-path backups\production-rehearsal.sql
```

See [MANAGED_DATA.md](MANAGED_DATA.md) for dry-run and `--execute` usage.

## Product Safeguards

The API exposes `GET /api/v1/safeguards`, and the frontend Responsible Use page displays the same launch-safe policy text.

Production signoff must confirm:

- Racing predictions are described as decision support, not betting advice.
- No product, page, README, or marketing copy claims guaranteed returns.
- Model limitations and data freshness are visible.
- The operator has the right to display each provider's race-card, odds, and result data.
- `RESPONSIBLE_GAMBLING_URL`, `PRIVACY_POLICY_URL`, and `TERMS_OF_USE_URL` point at approved public pages.
- Bet journal behavior is covered by the privacy policy before personal account-backed bet history is introduced.

## Launch Process

Staging acceptance:

```powershell
.\.venv\Scripts\python.exe scripts\production-readiness-check.py --base-url https://horse-predictor-api-staging.example.com --require-policy-links --require-approved-artifact
```

Production migration:

```powershell
$env:DATABASE_URL="mysql+pymysql://..."
.\scripts\staging-release.ps1
```

Initial historical import:

- Run `data_pipeline.py --provider <provider> --no-csv` against production.
- Keep the ingestion worker lock enabled.
- Confirm `api_ingestion_runs.status` records success.

Model approval:

- Review `/api/v1/model/evaluation`.
- Record a candidate snapshot and artifact with `POST /api/v1/admin/model/evaluation`.
- Require a non-empty chronological holdout.
- Compare model metrics against the market baseline before launch.
- Approve the reviewed model with `POST /api/v1/admin/model/{model_version_id}/approve`; approval verifies artifact checksum and feature-schema hash.
- Record a prediction snapshot with `POST /api/v1/admin/prediction-runs?require_approved_model=true`.
- Confirm the Admin Console audit history includes the snapshot, approval, and prediction-run actions.
- Record the approved commit SHA and data snapshot window with `scripts/release-record.py` or the staging acceptance workflow artifact.

When production is expected to have an approved model, add `--require-approved-model` to `scripts/production-readiness-check.py`.
When production is expected to serve from a persisted artifact, add `--require-approved-artifact`.
When production is expected to have persisted scoring evidence, add `--require-prediction-run` too.
When production is expected to have provider-grade enrichment coverage, add `--require-enriched-data`.
When production monitoring should be release-blocking, add `--require-monitoring --require-no-critical-alerts`.
When production admin governance should be release-blocking, add `--require-admin-governance` and provide an admin-capable account bearer token.

Monitored release:

- Deploy backend first and wait for `/api/v1/ready`.
- Deploy frontend with the production `VITE_API_BASE_URL`.
- Watch `/api/v1/monitoring`, JSON logs, and any platform telemetry for 4xx/5xx spikes, slow requests, drift, ingestion failures, stale freshness, and model-evaluation degradation.

Rollback:

- Roll frontend to the prior static build.
- Roll backend to the prior container image.
- Keep the database at the newest migrated schema unless a tested downgrade is available.
- Re-approve the previous artifact-backed model version if a model rollback is needed.
- If ingestion caused bad rows, pause the worker, restore from the latest clean backup, and re-run the production-readiness check.

## Agent Browser Gate

When `agent-browser` is available on PATH, run `scripts/visual-smoke-check.py --verbose` from `docs/TESTING.md` against the deployed frontend before marking launch acceptance complete.

See [CI_RELEASE.md](CI_RELEASE.md) for the full Phase 18 CI, staging acceptance, release-record, and visual gate workflow.
