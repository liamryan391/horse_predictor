# Repository Workflow

## Branches

- `main` is the stable production branch.
- `development` is the integration branch for accepted project work once it is created.
- Feature branches should use a short descriptive name such as `agent/add-api-react-platform`.
- Keep old migration branches until their contents are verified on `main` or `development`.

## Pull Requests

- Open a pull request for every non-trivial change.
- Use draft PRs for large platform changes until local validation passes.
- Keep PR descriptions focused on what changed, why it changed, and how it was tested.
- Do not merge work that exposes secrets, production database URLs, or provider credentials.
- For model changes, include the evaluation approach and whether leakage checks were run.

## Local Checks

Run backend syntax checks:

```powershell
.\.venv\Scripts\python.exe -m py_compile settings.py api.py data_pipeline.py horse_racing_app.py prediction_model.py racing_storage.py
```

Build the frontend:

```powershell
cd frontend
npm.cmd run build
```

Run ingestion with sample data:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe data_pipeline.py --provider sample
```

Provider ingestion changes should include adapter validation or a smoke test against a throwaway SQLite database.

Backend API changes should include contract smoke tests for response shape, filtering, and pagination.

## Release Flow

1. Merge feature work into `development`.
2. Run backend, frontend, ingestion, and model checks.
3. Promote `development` to `main` through a reviewed PR.
4. Tag important releases after the merge to `main`.
5. Keep rollback notes for schema or ingestion changes.
