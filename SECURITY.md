# Security Policy

## Reporting

Do not open public issues for suspected vulnerabilities, leaked credentials, or provider-access problems. Report them privately to the repository owner and include:

- affected commit or deployment URL
- reproduction steps
- observed impact
- any exposed request ID from the API response

## Sensitive Data

Never commit:

- `.env` files with real values
- database URLs with real usernames, passwords, hosts, or schema names
- racing provider API keys or account credentials
- exported production database backups
- user account, contact, payment, or bet-history records

Use `.env.example`, `.env.staging.example`, and `.env.production.example` only as checklists. Real values belong in the hosting platform secret store or a local untracked `.env` file.

## Supported Security Controls

- deployed runtime config validation through `Settings.validate_runtime()`
- CORS origin allowlisting with HTTPS enforcement outside local development
- trusted API host allowlisting through `ALLOWED_HOSTS`
- constant-time administrative bearer-token comparison
- request-size limiting through `MAX_REQUEST_BODY_BYTES`
- lightweight per-client API rate limiting
- JSON logs with request IDs for staging and production
- weekly Dependabot checks for Python, npm, and GitHub Actions dependencies
