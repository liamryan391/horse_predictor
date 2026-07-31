# Horse Predictor Roadmap

This roadmap is the working plan for turning Horse Predictor from a local prototype into a reliable racing intelligence platform. It preserves the planning context from the original project task and should be updated as phases are completed.

## Current Baseline

- Repository: `liamryan391/horse_predictor`
- Stable branch: `main`
- Current project branch: `agent/add-api-react-platform`
- Production database direction: MySQL
- Local/demo database fallback: SQLite
- Backend direction: FastAPI with SQLAlchemy
- Frontend direction: React and TypeScript
- Legacy diagnostic UI: Streamlit

## Phase 0: Repository Recovery And Governance

### 0.1 Remote Synchronization

- Fetch all GitHub branches and tags.
- Compare local and remote commit histories.
- Confirm whether `codex/create-horse-racing-software-and-app` contains the latest development work.
- Verify that no local-only commits are missing from GitHub.

Current inspection: `main` already includes the old Codex branch through PR #1. The active project branch is pushed to GitHub and open as PR #2.

### 0.2 Branch Organization

- Create `development` from the authoritative development commit.
- Push and track `origin/development`.
- Keep `main` as the stable production branch.
- Temporarily retain the old Codex branch until the rename is verified.
- Protect `main` from direct unreviewed changes in GitHub repository settings.

### 0.3 Repository Hygiene

- Add or verify `.gitignore`.
- Exclude virtual environments, database files, caches, build output, and secrets.
- Add `.env.example`.
- Document the pull-request and release workflow.
- Add issue and pull-request templates where useful.

## Phase 1: Architecture And Developer Experience

### 1.1 Confirm The Target Stack

Recommended target:

- React and TypeScript frontend.
- FastAPI Python backend.
- SQLAlchemy data-access layer.
- MySQL production database.
- SQLite only as a local/demo fallback.
- Alembic database migrations.
- Pytest backend tests.
- Vitest and React Testing Library frontend tests.

Current inspection: the stack is documented and the repo now has FastAPI, SQLAlchemy, React/TypeScript, MySQL-ready config, SQLite fallback, and Streamlit diagnostics. Alembic and test frameworks remain Phase 2/7 work.

### 1.2 Local Environment

- Add Windows PowerShell setup instructions.
- Add Linux/macOS setup instructions.
- Add VS Code interpreter configuration.
- Add one-command local startup where practical.
- Add Docker Compose for backend, frontend, and MySQL.

Current inspection: `docs/SETUP.md`, `scripts/start-dev.ps1`, `scripts/start-dev.sh`, and `docker-compose.yml` cover these paths.

### 1.3 Configuration

- Centralize environment-variable handling.
- Validate required production settings at startup.
- Separate development, test, staging, and production configurations.
- Ensure secrets are never committed.

Current inspection: `settings.py` centralizes environment reads, loads `.env`, validates staging/production runtime settings, and keeps secrets out of committed examples.

## Phase 2: SQL Data Platform

### 2.1 Relational Schema

Create normalized tables for:

- Race meetings.
- Individual races.
- Race entries and runners.
- Horses.
- Jockeys.
- Trainers.
- Owners.
- Courses/tracks.
- Historical results.
- Odds snapshots.
- User bets.
- API ingestion runs.
- Prediction runs.
- Model versions.
- Model evaluation results.

### 2.2 Data Integrity

- Add primary keys and foreign keys.
- Add unique constraints for provider race/runner identifiers.
- Add indexes for dates, tracks, horses, jockeys, and races.
- Use UTC ingestion timestamps.
- Preserve source-provider identifiers.
- Add raw payload storage or references for traceability.

### 2.3 Migrations

- Initialize Alembic.
- Add an initial schema migration.
- Add seed-data commands.
- Test upgrades and downgrades.
- Document backup and restore procedures.

### 2.4 Data-Access Layer

- Replace direct SQLite-specific code with SQLAlchemy repositories.
- Add transaction handling.
- Use safe upserts rather than replacing whole tables.
- Add pagination and filtered queries.
- Add database health reporting.

Current inspection: Alembic migration scaffolding, normalized schema tables, and provider upsert writes are in place. Sample ingestion still uses replace mode for repeatable local demos.

## Phase 3: Racing API Ingestion

### 3.1 Provider Selection

Evaluate providers based on:

- Historical coverage.
- Upcoming race cards.
- Results.
- Jockey, owner, trainer, and horse identifiers.
- Odds history.
- Rate limits.
- UK/Ireland and international coverage.
- Licensing and permitted commercial use.
- Cost.

### 3.2 Provider Adapter

- Implement a provider interface.
- Separate provider payloads from internal database models.
- Add typed schema validation.
- Normalize dates, distances, surfaces, going, odds, and identities.
- Handle pagination.

Current inspection: provider mapping now lives in `provider_adapters.py` with adapter classes for `generic`, `theracingapi`, and `ourhub`. Provider rows are normalized and validated before database writes, and The Racing API result fetching has bounded pagination support.

### 3.3 Reliability

- Add retries with exponential backoff.
- Respect rate limits.
- Add timeouts.
- Make ingestion idempotent.
- Record success/failure and row counts.
- Preserve partial progress safely.
- Alert on repeated failures.

Current inspection: remote provider calls now support timeout, retry, backoff, minimum request interval, and failure logging to `api_ingestion_runs`. Provider writes use Phase 2 upserts. Alerting remains future work.

### 3.4 Scheduling

- Start with a local manual command.
- Add scheduled staging ingestion.
- Refresh upcoming races hourly.
- Refresh odds more frequently only if licensing and provider limits allow it.
- Import official results after races.
- Reconcile postponed, abandoned, and corrected races.

Current inspection: local manual commands and hourly loop remain available. A proper hosted scheduler or worker remains Phase 4/8 work.

## Phase 4: Prediction And Evaluation

### 4.1 Eliminate Leakage

- Ensure result-only fields such as `finishing_position` never become model inputs.
- Keep target creation separate from feature creation.
- Add explicit leakage tests.

Current inspection: `prediction_model.train_model` excludes `finishing_position`, `is_winner`, and `race_date` from predictors, and `tests/test_prediction_model.py` now covers the leakage guard.

### 4.2 Feature Engineering

Potential features include:

- Recent horse form.
- Course and distance performance.
- Going/surface performance.
- Days since last run.
- Official and speed ratings.
- Weight carried.
- Draw.
- Race class.
- Jockey strike rate.
- Trainer strike rate.
- Jockey/trainer partnership.
- Owner trends.
- Market-implied probability.
- Odds movement.
- Field size.

Current inspection: the shared model module now adds pre-race context features for implied probability, field size, odds rank, relative speed rating, and relative class rating.

### 4.3 Proper Validation

- Use chronological train/validation/test splits.
- Avoid random leakage across future and past races.
- Evaluate by race rather than only by runner.
- Measure log loss, Brier score, calibration, ranking quality, and top-pick win rate.
- Compare against market odds as a baseline.

Current inspection: `evaluate_model` now uses chronological race-level holdout validation and reports runner log loss, Brier score, calibration error, ranking quality, top-pick win rate, and market baseline metrics through `/api/model` and `/api/model/evaluation`.

### 4.4 Backtesting

- Simulate fixed-stake and proportional-stake strategies.
- Account for overround and commission.
- Report return on investment and drawdown.
- Prevent retrospective use of unavailable information.
- Clearly separate predictive performance from betting profitability.

Current inspection: the evaluator now reports a basic fixed-stake value-bet profit and ROI separately from predictive metrics. Drawdown, commission, and proportional staking remain future work.

### 4.5 Model Lifecycle

- Store model version and training range.
- Save evaluation metrics.
- Promote models only if they outperform the current approved model.
- Add retraining schedules.
- Never describe automatic retraining as "self-improvement" unless promotion is governed by validated metrics.

Current inspection: model metrics are visible in the API, React Race Lab, and Streamlit model tab. Phase 12 now persists approved model artifacts for serving. Promotion gates and retraining schedules remain future work.

## Phase 5: Backend API

### 5.1 Core Endpoints

- Health and readiness.
- Meetings and races.
- Race-card runners.
- Horse profiles and form.
- Jockey, trainer, and owner profiles.
- Predictions.
- Trends.
- Model status.
- Ingestion status.
- User bet history.

### 5.2 API Quality

- Add Pydantic response models.
- Add pagination.
- Add sorting and filtering.
- Standardize error responses.
- Add API versioning.
- Generate OpenAPI documentation.
- Add request IDs and structured logging.

### 5.3 Security

- Add authentication.
- Add role-based authorization where needed.
- Restrict administrative ingestion/model endpoints.
- Add rate limiting.
- Lock down CORS.
- Avoid exposing database URLs or API credentials in responses.

Current inspection: `/api/health` and `/api/summary` return sanitized database metadata instead of the raw database URL.

Current inspection: backend routes are now mounted under `/api/v1` with `/api` compatibility, Pydantic response models, request IDs, standardized error envelopes, filtering, sorting, pagination, meetings/races/entity-profile endpoints, API docs, and a protected sample-seed admin endpoint. The frontend now calls the versioned API. Full user auth, role-based authorization, and production gateway rate limiting remain future work.

## Phase 6: Professional Frontend

### 6.1 Public Pages

- Home.
- About Us.
- Methodology.
- Responsible Use.
- Contact/support.

Current inspection: the React app now opens to the operational workspace and includes Methodology, Responsible Use, and Contact pages.

### 6.2 Racing Intelligence Workspace

- Today's meetings.
- Race-card browser.
- Runner comparison.
- Prediction rankings.
- Probability and model-odds views.
- Confidence and data-quality indicators.
- Horse, jockey, trainer, and owner details.

Current inspection: the workspace now uses `/api/v1` for meetings, race-card rows, prediction rankings, runner comparison, trend tables, model evaluation, saved track filters, and data-quality indicators.

### 6.3 Bet Journal

- Record a bet.
- Import bet history.
- Track settled and unsettled bets.
- Show profit/loss and return on investment.
- Filter by track, bet type, horse, and date.
- Avoid language that guarantees profit.

Current inspection: the frontend now includes a browser-local bet journal for open/settled positions, stake, profit, and ROI. Server-side bet history remains future work.

### 6.4 Dynamic Experience

- Responsive navigation.
- Loading skeletons.
- Empty states.
- Error recovery.
- Accessible tables and controls.
- Dark mode.
- Saved filters.
- Charts for form, odds movements, and model calibration.

Current inspection: the frontend now has responsive workspace navigation, loading skeletons, retryable error states, saved filters, dark mode, and a model calibration bar. Full accessibility automation and richer charts remain Phase 7+ work.

## Phase 7: Automated Testing

### 7.1 Backend

- Unit tests for transformations.
- API contract tests.
- Database repository tests.
- Migration tests.
- Provider adapter tests with recorded fixtures.
- Model leakage tests.
- Prediction determinism tests.

Current inspection: the Python suite now covers API contracts, provider fixture flattening, provider validation summaries, SQLite repository upserts, Alembic upgrade/downgrade roundtrips, leakage checks, chronological holdout behavior, and deterministic sample-data scoring gates.

### 7.2 Frontend

- Component tests.
- Navigation tests.
- API error-state tests.
- Accessibility checks.
- Responsive-layout checks.
- End-to-end race-card workflow.

Current inspection: the frontend now has Node test coverage for browser-local bet journal math, draft validation, and stored-row parsing. Vite build and live HTTP/browser smoke checks remain the UI gate until a dedicated component/e2e runner is introduced.

### 7.3 Data And Model Tests

- Required-column checks.
- Invalid odds and result checks.
- Duplicate race/runner checks.
- Chronological split checks.
- Calibration thresholds.
- Baseline comparison.

Current inspection: `data_quality.py` now provides reusable required-column, invalid odds/result, duplicate race/runner, calibration, and market-baseline checks used by the automated test suite.

## Phase 8: Cloud Staging

### 8.1 Deployment

- Deploy frontend static assets.
- Deploy FastAPI backend.
- Provision managed MySQL.
- Configure secrets.
- Run migrations.
- Add HTTPS and domain configuration.

Current inspection: staging now has a managed-database env template, staging compose manifest, backend release scripts, a production-style static frontend container, and cloud staging runbook. Actual hosted MySQL, HTTPS, and domain provisioning remain external platform work.

### 8.2 Scheduled Jobs

- Deploy ingestion worker.
- Add job locking to prevent duplicate runs.
- Add retry queues.
- Add alerts.
- Add data-freshness indicators in the UI.

Current inspection: the ingestion worker now supports a database-backed `job_locks` table to skip overlapping runs, and the workspace shows API-provided data freshness from `/api/v1/summary`.

### 8.3 Observability

- Structured application logs.
- Error tracking.
- Performance monitoring.
- Database metrics.
- Ingestion failure dashboards.
- Model-quality monitoring.

Current inspection: staging can emit JSON application logs, and the staging runbook identifies health, readiness, ingestion, freshness, and model-evaluation checks. Full dashboards and external alert routing remain future platform configuration.

## Phase 9: Production Readiness

### 9.1 Security Review

- Secret rotation.
- Dependency scanning.
- Database least-privilege accounts.
- Authentication review.
- Backup restoration test.
- API abuse protections.

Current inspection: deployed runtime validation now requires managed SQL, HTTPS CORS origins, explicit API host allowlisting, and a strong admin token; API responses get baseline security headers, request IDs are sanitized, request body size is limited, admin tokens use constant-time comparison, and Dependabot is configured for Python/npm/GitHub Actions. Secret rotation, database roles, external rate limiting, and backup restore execution remain operator tasks.

### 9.2 Product Safeguards

- Responsible gambling notice.
- Clear uncertainty and model limitations.
- No guaranteed-return claims.
- Licensing review for displayed data.
- Privacy policy for user accounts and bet history.
- Terms of use.

Current inspection: `/api/v1/safeguards` and the frontend Responsible Use page now surface responsible-use, model-limit, data-licensing, privacy, terms, and optional policy-link notices. Final policy URLs, provider display rights, and legal approval remain required before production launch.

### 9.3 Launch Process

- Staging acceptance test.
- Production migration.
- Initial historical-data import.
- Model approval.
- Monitored release.
- Rollback plan.

Current inspection: `docs/PRODUCTION_READINESS.md`, `.env.production.example`, and `scripts/production-readiness-check.py` now define the launch acceptance, migration, initial import, model approval, monitoring, and rollback gates. Real production migration, historical import, and monitored release remain future platform execution.

## Phase 10: Model Operations

### 10.1 Model Registry

- Persist model version metadata.
- Persist evaluation metrics for each candidate.
- Expose a registry API endpoint.
- Show model registry status in the frontend.

Current inspection: model evaluation snapshots can now be recorded into `model_versions` and `model_evaluation_results`, `/api/v1/model/registry` lists candidate/approved/superseded rows, and the React Evaluation view displays the latest registry entries.

### 10.2 Model Approval

- Require explicit model approval before production launch.
- Keep previous approved models auditable.
- Add an acceptance check for approved models.

Current inspection: `POST /api/v1/admin/model/{model_version_id}/approve` marks one model approved and supersedes any previous approved row. `scripts/production-readiness-check.py --require-approved-model` can enforce this gate for staging or production.

### 10.3 Future Model Lifecycle

- Compare candidates against the current approved model before promotion.
- Add scheduled retraining with guarded promotion.

Current inspection: Phase 10 records model metadata and metrics. Phase 12 now persists trained artifacts and serves from approved artifacts, while candidate-vs-approved comparison and scheduled retraining remain future work.

## Phase 11: Prediction Operations

### 11.1 Prediction Run Persistence

- Store prediction-run headers.
- Store runner-level prediction rows for each scored race card.
- Link prediction runs to the latest approved model metadata when available.
- Preserve model probabilities, model odds, value edge, and suggested rank.

Current inspection: `prediction_runs` now has runner-level `prediction_run_entries`. `POST /api/v1/admin/prediction-runs` records the current scored race card and links it to the latest approved model metadata when available.

### 11.2 Prediction Run API And UI

- Expose recent prediction runs through the API.
- Expose prediction-run detail rows through the API.
- Show recent prediction snapshots in the React Evaluation view.
- Add a staging/production acceptance check for persisted prediction evidence.

Current inspection: `/api/v1/prediction-runs` lists run summaries, `/api/v1/prediction-runs/{prediction_run_id}` returns runner-level entries, the React Evaluation view shows recent runs, and `scripts/production-readiness-check.py --require-prediction-run` can enforce the gate.

### 11.3 Future Prediction Lifecycle

- Add scheduled prediction snapshot jobs after ingestion refreshes.
- Add drift checks comparing live prediction distributions with holdout expectations.
- Add prediction-run cleanup or archival policies once production volume is known.

Current inspection: Phase 11 persists scoring snapshots. Phase 12 now links prediction-run capture to approved artifact serving when required, while scheduled scoring, drift monitoring, and retention policies remain future work.

## Continued Planning

The next build plan continues in `ROADMAP02.md`. Phase 12: Artifact-Backed Model Serving is implemented; the next planned stage is Phase 13: Data Enrichment And Provider Depth.

Longer-term product, data, modelling, governance, and monitoring ideas are captured in `FUTURE_ROADMAP.md`.
