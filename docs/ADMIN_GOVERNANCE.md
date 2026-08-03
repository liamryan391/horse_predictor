# Admin Console And Governance

Phase 17 adds a lightweight operator governance layer for model and prediction operations. It is intentionally bearer-token based for now; a later authentication phase should replace shared tokens with named user accounts, stronger role management, passwordless sign-in, and per-user journal ownership.

## Access Model

Local development keeps the app easy to run:

- If `API_AUTH_TOKEN` and `JOURNAL_AUTH_TOKEN` are both empty outside staging/production, admin and journal routes allow a `local-dev` operator context.
- If either token is configured, callers should send `Authorization: Bearer <token>`.
- Admin tokens grant `reader`, `journal`, and `admin` roles.
- Journal tokens grant `reader` and `journal` roles only.
- `X-Admin-Actor` or `X-Journal-Actor` can label the operator in audit rows.

Staging and production require:

- `API_AUTH_TOKEN`
- `JOURNAL_AUTH_TOKEN`
- `JOURNAL_ACCOUNT_KEY`

Both tokens must be non-placeholder secrets with at least 32 characters.

## Admin Console

The React Admin tab stores the access token and actor label in browser local storage. It calls:

- `GET /api/v1/admin/session`
- `GET /api/v1/admin/governance`
- `GET /api/v1/admin/audit-log`

It also exposes guarded actions:

- capture a model evaluation snapshot and artifact
- approve a model version for artifact-backed serving
- supersede a model version
- capture a prediction-run snapshot with `require_approved_model=true`
- seed sample data

Every action asks for confirmation before the API call.

## Audit Log

Governed admin writes append rows to `admin_audit_events`.

Each event stores:

- actor
- roles
- action
- resource type and id
- request id
- status
- detail
- compact JSON payload
- creation timestamp

The audit log is meant for operator traceability. It is not a compliance-grade immutable ledger yet.

## Journal Scope

The server-side bet journal now reads the account key from the access context:

- local development uses `local`
- an admin token uses `admin`
- a journal token uses `JOURNAL_ACCOUNT_KEY`

This keeps Phase 15 journal records server-backed while preparing for real account ownership later.

## Production Checks

The readiness helper can validate the admin governance surface:

```powershell
$env:API_AUTH_TOKEN="your-32-character-admin-token"
.\.venv\Scripts\python.exe scripts\production-readiness-check.py --base-url https://horse-predictor-api-staging.example.com --require-admin-governance
```

Use the admin check alongside the existing approved-artifact, prediction-run, enrichment, and monitoring gates before wider release.
