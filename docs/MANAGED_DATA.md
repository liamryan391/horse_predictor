# Managed Data And Hosting

Phase 23 prepares Horse Predictor for a real staging release with managed SQL, hosted secrets, backup/restore rehearsal, and acceptance evidence.

## Managed Database

Staging and production should use MySQL or MariaDB through `DATABASE_URL`. SQLite remains local-only.

Required database boundaries:

- Use a managed database service outside the app container.
- Use separate credentials for app runtime, migrations, backups, and ingestion jobs where the host allows it.
- Keep production, staging, and restore-test databases separate.
- Never restore a backup into the same host/port/database identity that produced it.
- Give restore-test database names an obvious suffix such as `_restore`, `_staging`, `_test`, `_sandbox`, or `_rehearsal`.

## Environment Check

Check staged environment values before deployment:

```powershell
.\.venv\Scripts\python.exe scripts\staging-env-check.py --env-file .env.staging.example --allow-placeholders --require-release-token
```

For a real secret export, remove `--allow-placeholders` and add stricter gates:

```powershell
.\.venv\Scripts\python.exe scripts\staging-env-check.py --env-file .env.staging --require-release-token --require-policy-links --require-provider-credentials
```

The checker validates managed SQL, HTTPS CORS/frontend URLs, allowed hosts, account auth, release-helper token availability, policy links, provider credentials, and placeholder values. Sensitive values are redacted in the JSON report.

## Backup And Restore Rehearsal

Dry-run a backup/restore plan first:

```powershell
.\.venv\Scripts\python.exe scripts\managed-db-rehearsal.py `
  --mode roundtrip `
  --source-url $env:DATABASE_URL `
  --restore-url $env:RESTORE_DATABASE_URL `
  --backup-path backups\staging-rehearsal.sql
```

Run the rehearsal only after confirming the source and restore targets:

```powershell
.\.venv\Scripts\python.exe scripts\managed-db-rehearsal.py `
  --mode roundtrip `
  --source-url $env:DATABASE_URL `
  --restore-url $env:RESTORE_DATABASE_URL `
  --backup-path backups\staging-rehearsal.sql `
  --execute
```

The script uses `mysqldump` and `mysql` from `PATH`, passes database passwords through `MYSQL_PWD`, and never prints raw passwords. `backups/` is ignored by git.

## Migration And Acceptance

After restore, point `DATABASE_URL` at the restored database and run:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe scripts\production-readiness-check.py --base-url $env:API_BASE_URL --require-approved-artifact --require-monitoring --require-admin-governance --admin-token $env:ACCOUNT_AUTH_TOKEN
.\.venv\Scripts\python.exe scripts\release-record.py --base-url $env:API_BASE_URL --admin-token $env:ACCOUNT_AUTH_TOKEN --output release-records\staging-release-record.json
```

For GitHub-hosted acceptance, run `.github/workflows/staging-acceptance.yml` manually with the staging API URL and an `ACCOUNT_AUTH_TOKEN` repository secret.

## Rollback Boundaries

Treat migrations as forward-only unless a downgrade has been tested against a restored copy.

Safe rollback order:

- Roll frontend to the previous static build.
- Roll backend to the previous image.
- Keep the database on the newest migrated schema unless the restore rehearsal proved downgrade safety.
- Re-approve the previous model artifact if model serving needs a rollback.
- If ingestion polluted provider rows, pause the worker and restore from the latest clean backup into a new database before switching traffic.
