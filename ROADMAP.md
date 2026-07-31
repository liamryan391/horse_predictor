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

### 3.3 Reliability

- Add retries with exponential backoff.
- Respect rate limits.
- Add timeouts.
- Make ingestion idempotent.
- Record success/failure and row counts.
- Preserve partial progress safely.
- Alert on repeated failures.

### 3.4 Scheduling

- Start with a local manual command.
- Add scheduled staging ingestion.
- Refresh upcoming races hourly.
- Refresh odds more frequently only if licensing and provider limits allow it.
- Import official results after races.
- Reconcile postponed, abandoned, and corrected races.

## Phase 4: Prediction And Evaluation

### 4.1 Eliminate Leakage

- Ensure result-only fields such as `finishing_position` never become model inputs.
- Keep target creation separate from feature creation.
- Add explicit leakage tests.

Current inspection: `prediction_model.train_model` excludes `finishing_position`, `is_winner`, and `race_date` from predictors. This is good, but it needs a regression test.

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

### 4.3 Proper Validation

- Use chronological train/validation/test splits.
- Avoid random leakage across future and past races.
- Evaluate by race rather than only by runner.
- Measure log loss, Brier score, calibration, ranking quality, and top-pick win rate.
- Compare against market odds as a baseline.

### 4.4 Backtesting

- Simulate fixed-stake and proportional-stake strategies.
- Account for overround and commission.
- Report return on investment and drawdown.
- Prevent retrospective use of unavailable information.
- Clearly separate predictive performance from betting profitability.

### 4.5 Model Lifecycle

- Store model version and training range.
- Save evaluation metrics.
- Promote models only if they outperform the current approved model.
- Add retraining schedules.
- Never describe automatic retraining as "self-improvement" unless promotion is governed by validated metrics.

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

## Phase 6: Professional Frontend

### 6.1 Public Pages

- Home.
- About Us.
- Methodology.
- Responsible Use.
- Contact/support.

### 6.2 Racing Intelligence Workspace

- Today's meetings.
- Race-card browser.
- Runner comparison.
- Prediction rankings.
- Probability and model-odds views.
- Confidence and data-quality indicators.
- Horse, jockey, trainer, and owner details.

### 6.3 Bet Journal

- Record a bet.
- Import bet history.
- Track settled and unsettled bets.
- Show profit/loss and return on investment.
- Filter by track, bet type, horse, and date.
- Avoid language that guarantees profit.

### 6.4 Dynamic Experience

- Responsive navigation.
- Loading skeletons.
- Empty states.
- Error recovery.
- Accessible tables and controls.
- Dark mode.
- Saved filters.
- Charts for form, odds movements, and model calibration.

## Phase 7: Automated Testing

### 7.1 Backend

- Unit tests for transformations.
- API contract tests.
- Database repository tests.
- Migration tests.
- Provider adapter tests with recorded fixtures.
- Model leakage tests.
- Prediction determinism tests.

### 7.2 Frontend

- Component tests.
- Navigation tests.
- API error-state tests.
- Accessibility checks.
- Responsive-layout checks.
- End-to-end race-card workflow.

### 7.3 Data And Model Tests

- Required-column checks.
- Invalid odds and result checks.
- Duplicate race/runner checks.
- Chronological split checks.
- Calibration thresholds.
- Baseline comparison.

## Phase 8: Cloud Staging

### 8.1 Deployment

- Deploy frontend static assets.
- Deploy FastAPI backend.
- Provision managed MySQL.
- Configure secrets.
- Run migrations.
- Add HTTPS and domain configuration.

### 8.2 Scheduled Jobs

- Deploy ingestion worker.
- Add job locking to prevent duplicate runs.
- Add retry queues.
- Add alerts.
- Add data-freshness indicators in the UI.

### 8.3 Observability

- Structured application logs.
- Error tracking.
- Performance monitoring.
- Database metrics.
- Ingestion failure dashboards.
- Model-quality monitoring.

## Phase 9: Production Readiness

### 9.1 Security Review

- Secret rotation.
- Dependency scanning.
- Database least-privilege accounts.
- Authentication review.
- Backup restoration test.
- API abuse protections.

### 9.2 Product Safeguards

- Responsible gambling notice.
- Clear uncertainty and model limitations.
- No guaranteed-return claims.
- Licensing review for displayed data.
- Privacy policy for user accounts and bet history.
- Terms of use.

### 9.3 Launch Process

- Staging acceptance test.
- Production migration.
- Initial historical-data import.
- Model approval.
- Monitored release.
- Rollback plan.
