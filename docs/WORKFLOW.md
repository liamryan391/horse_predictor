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
- Confirm the GitHub Actions CI workflow is passing before marking a PR ready for review.
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

UI changes should include a Vite production build and, when a preview is available, `scripts/visual-smoke-check.py`.

## Release Flow

1. Merge feature work into `development`.
2. Confirm CI passes on the `development` PR or branch.
3. Run staging acceptance with `scripts/staging-release.ps1`, `scripts/staging-release.sh`, or the manual GitHub Actions staging acceptance workflow.
4. Store the release record artifact with the deployment notes.
5. Promote `development` to `main` through a reviewed PR.
6. Tag important releases after the merge to `main`.
7. Keep rollback notes for schema or ingestion changes.

See [CI_RELEASE.md](CI_RELEASE.md) for the Phase 18 CI, staging acceptance, release-record, and visual gate workflow.
