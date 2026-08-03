# Admin Console And Governance

Phase 17 added a lightweight operator governance layer for model and prediction operations. Phase 22 upgrades that layer with named operator accounts, hashed account bearer tokens, role-aware access checks, and account-owned journal scope.

## Access Model

Local development keeps the app easy to run:

- If `API_AUTH_TOKEN` and `JOURNAL_AUTH_TOKEN` are both empty outside staging/production, admin and journal routes allow a `local-dev` operator context.
- Preferred local auth is a named `operator_accounts` row with a hashed token.
- Legacy `API_AUTH_TOKEN` and `JOURNAL_AUTH_TOKEN` still work as local-development fallback tokens.
- `X-Account-Actor`, `X-Admin-Actor`, or `X-Journal-Actor` can label the operator in audit rows.

Create or update an operator account:

```powershell
.\.venv\Scripts\python.exe scripts\account-smoke-check.py --database-url horse_racing.db --account-key liam --display-name "Liam" --roles admin,journal --token "replace-with-a-long-local-token" --privacy-acknowledged
```

Account roles:

- `viewer`: read-only app/session access.
- `journal`: read and write the account-owned bet journal.
- `operator`: journal plus operational actions that future ingestion controls can use.
- `admin`: model release, governance, account management, and all lower roles.
- `release-approver`: reserved for future two-person release checks.

Staging and production require:

- `ACCOUNT_AUTH_ENABLED=true`
- at least one admin-capable `operator_accounts` row
- `JOURNAL_ACCOUNT_KEY`

Legacy shared tokens should be kept for local/dev only. If `ACCOUNT_AUTH_ENABLED=false` is deliberately used for a private deployed rehearsal, `API_AUTH_TOKEN` and `JOURNAL_AUTH_TOKEN` must be non-placeholder secrets with at least 32 characters.

## Admin Console

The React Admin tab stores the access token and actor label in browser local storage. It calls:

- `GET /api/v1/auth/session`
- `GET /api/v1/admin/session`
- `GET /api/v1/admin/accounts`
- `GET /api/v1/admin/governance`
- `GET /api/v1/admin/audit-log`

It also exposes guarded actions:

- capture a model evaluation snapshot and artifact
- approve a model version for artifact-backed serving
- supersede a model version
- capture a prediction-run snapshot with `require_approved_model=true`
- create or update named operator accounts through the API
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

- account tokens use the matching `operator_accounts.account_key`
- local development fallback uses `local`
- a legacy admin token uses `admin`
- a legacy journal token uses `JOURNAL_ACCOUNT_KEY`

This keeps Phase 15 journal records server-backed and gives each named account its own journal scope. Export/delete controls and privacy-policy enforcement are still future product work before broad personal-use rollout.

## Production Checks

The readiness helper can validate the admin governance surface:

```powershell
$env:ACCOUNT_AUTH_TOKEN="your-admin-account-token"
.\.venv\Scripts\python.exe scripts\production-readiness-check.py --base-url https://horse-predictor-api-staging.example.com --require-admin-governance
```

For account-token checks, pass the account token wherever the script expects an admin token. Use the admin check alongside the existing approved-artifact, prediction-run, enrichment, and monitoring gates before wider release.
