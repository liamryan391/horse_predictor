# Model Operations

Phase 10 adds an auditable model registry foundation. Phase 12 extends it with artifact-backed serving: candidate snapshots now write a serialized model artifact, approval verifies artifact integrity, and prediction routes can serve from the approved artifact instead of retraining on every request. Phase 17 records governed model actions in the admin audit log.

## Model Registry

Public registry endpoint:

```text
GET /api/v1/model/registry
```

The response includes:

- model version id
- name
- algorithm
- status: `candidate`, `approved`, or `superseded`
- training date window
- feature count
- artifact URI, artifact checksum, feature-schema hash, and code commit SHA when a persisted artifact exists
- persisted evaluation metrics
- created and updated timestamps

The React Evaluation view shows the latest registry rows below the live model evaluation metrics, including whether each snapshot has a tracked artifact.

## Recording A Candidate Snapshot

Use the administrative endpoint after a fresh data import or before release review:

```powershell
Invoke-WebRequest `
  -Method POST `
  -Headers @{ Authorization = "Bearer $env:ACCOUNT_AUTH_TOKEN"; "X-Account-Actor" = "Liam" } `
  -UseBasicParsing `
  "http://127.0.0.1:8000/api/v1/admin/model/evaluation"
```

This trains the current model, writes a serialized artifact under `MODEL_ARTIFACT_DIR`, runs chronological holdout evaluation, stores a `candidate` row in `model_versions`, and stores non-empty metrics in `model_evaluation_results`.

Artifact metadata recorded with the registry row includes:

- `artifactUri`
- `artifactSha256`
- `featureSchemaHash`
- `codeCommitSha`

Local development writes artifacts to `model_artifacts/` by default. The directory is ignored by git. Staging and production should point `MODEL_ARTIFACT_DIR` at persistent storage. The API only loads registry artifacts from this configured directory.

## Approving A Model

Approve a candidate only after reviewing the metrics against the market baseline and checking the data window:

```powershell
Invoke-WebRequest `
  -Method POST `
  -Headers @{ Authorization = "Bearer $env:ACCOUNT_AUTH_TOKEN"; "X-Account-Actor" = "Liam" } `
  -UseBasicParsing `
  "http://127.0.0.1:8000/api/v1/admin/model/1/approve"
```

Approval now loads the candidate artifact and verifies the stored checksum and feature-schema hash before changing status. Approving one model marks any previous `approved` model as `superseded`. Rollback uses the same endpoint by approving a prior artifact-backed model version.

## Superseding A Model

Use supersede when a candidate or previously approved model should be removed from active consideration without deleting registry or audit history:

```powershell
Invoke-WebRequest `
  -Method POST `
  -Headers @{ Authorization = "Bearer $env:ACCOUNT_AUTH_TOKEN"; "X-Account-Actor" = "Liam" } `
  -UseBasicParsing `
  "http://127.0.0.1:8000/api/v1/admin/model/1/supersede"
```

The Admin Console exposes the same action and writes an `admin_audit_events` row.

## Launch Gate

Production acceptance can require an approved model artifact:

```powershell
.\.venv\Scripts\python.exe scripts\production-readiness-check.py `
  --base-url https://horse-predictor-api.example.com `
  --require-policy-links `
  --require-approved-model `
  --require-approved-artifact `
  --require-hsts
```

For staging and production, `REQUIRE_APPROVED_MODEL_ARTIFACT=true` makes `/api/v1/ready` fail until an approved artifact can be loaded.

## Current Limits

- There is no automated model promotion; approval remains an explicit administrative action.
- Shared bearer-token roles should be replaced by named user accounts before broad operator access.
- Candidate-vs-approved metric comparison and scheduled retraining remain future work.
- Artifact storage is local filesystem based; production should mount durable storage or map the artifact path to managed object storage in a later phase.

See [PREDICTION_OPERATIONS.md](PREDICTION_OPERATIONS.md) for persisted scoring snapshots.
