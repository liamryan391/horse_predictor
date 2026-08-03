# Roadmap03

Roadmap02 is complete through Phase 18. Roadmap03 moves the project from a strong local/staging platform toward live-provider certification, account-owned usage, production hosting, and better race-day operation.

## Current Release Check

August 3, 2026 pre-merge checks found:

- local CI, API, DB, model, release-record, and frontend build paths pass
- internet access works from the machine
- Open-Meteo provider-depth checks work for course enrichment
- OurHub and The Racing API endpoints are reachable but require credentials
- no local `.env` or process environment currently provides racing-provider credentials
- sample ingestion writes rows into a fresh DB, API track matching works for York, and prediction ranking works
- `agent-browser` can become daemon-blocked on this machine, so browser automation should be stabilized before it becomes the only visual release gate

## Internet And Provider Leads

Checked source leads:

- OurHub Racing API: documents `course-info/{race_date}` and `runner-info/{race_date}` endpoints with `X-API-Key` authentication. Source: https://github.com/TamB10/ourhub-racing-api
- The Racing API: supports racecards/results and requires account credentials. Source: https://api.theracingapi.com/documentation
- BSD Horse Racing API: documents tracks, meetings, races, live races, next-to-jump races, runners, and odds endpoints. Source: https://sports.bzzoiro.com/docs/horseracing/
- FormFav: documents racing data feed, meetings, race form, and predictions resources with a free-tier claim. Source: https://formfav.com/docs
- Open-Meteo historical weather remains useful for no-secret course/weather enrichment checks. Source: https://open-meteo.com/en/docs/historical-weather-api

## Phase 19: Live Provider Certification

Goal: prove authenticated race-card and result imports from the chosen provider.

### 19.1 Credentials And Secret Handling

- Configure `.env` locally and hosting secrets in staging.
- Validate OurHub and The Racing API credentials without printing secrets.
- Add a provider smoke command that redacts credentials and reports endpoint, status, rows, and validation issues.

### 19.2 Live Race-Card Import

- Import today or next available race cards into a throwaway DB.
- Verify meetings, track filters, race-card rows, prediction rows, data quality, and ingestion status.
- Document which provider fields map into `track`, `distance`, `surface`, `horse`, `jockey`, `trainer`, `owner`, `odds`, and draw/ratings fields.

### 19.3 Historical Results Import

- Import enough historical rows to make holdout metrics meaningful.
- Confirm provider result fields include finishing positions and stable race/runner identifiers.
- Add freshness and minimum-history gates for live-provider acceptance.

### 19.4 Track Alias Normalization

- Add provider course-name alias mapping.
- Treat `York`, `York Racecourse`, and provider-specific spelling as one track.
- Expose alias/provenance warnings in data quality.

## Phase 20: Provider Entity Model

Goal: replace flat natural-key upserts with stable provider entities.

### 20.1 Stable Provider IDs

- Store provider race IDs, runner IDs, horse IDs, jockey IDs, trainer IDs, owner IDs, and odds timestamps when available.
- Keep source provider and fetched-at timestamps on every imported entity.
- Preserve current flat compatibility tables until the frontend/model migrates cleanly.

### 20.2 Normalized Read Models

- Populate `race_meetings`, `races`, `race_entries`, `historical_results`, and `odds_snapshots`.
- Add API endpoints that read from normalized tables while maintaining current `/race-card` compatibility.
- Add migration/backfill checks before switching model features to normalized data.

### 20.3 Provider Comparison Matrix

- Compare OurHub, The Racing API, BSD, FormFav, Betfair, and any official/licensed racing partners.
- Track coverage, cost, authentication, quotas, racecard depth, results depth, odds depth, stable IDs, live latency, and redistribution terms.

## Phase 21: Account Authentication And User Ownership

Goal: replace shared bearer tokens with named users and account-owned data.

### 21.1 Auth Provider

- Add login/session management.
- Support viewer, journal, operator, admin, and release-approver roles.
- Keep bearer-token fallback for local development only.

### 21.2 Account-Owned Bet Journal

- Scope journal rows to real user accounts.
- Add export/delete controls.
- Add privacy-policy checks before storing personal betting history.

### 21.3 Governance Audit Upgrade

- Record auth events, provider config changes, ingestion backfills, release records, and rollback actions.
- Add actor identity to every governed action.

## Phase 22: Production Hosting And Managed Data

Goal: deploy a release candidate with managed database, secrets, backups, and release checks.

### 22.1 Managed Database

- Move staging to managed MySQL or another managed SQL database.
- Prove backup and restore into an isolated database.
- Add migration runbook and rollback boundaries.

### 22.2 Secret And Runtime Configuration

- Store provider credentials, admin tokens, journal tokens, and database URLs in hosting secrets only.
- Configure CORS, allowed hosts, policy links, data license references, and artifact storage.

### 22.3 Staging Acceptance

- Run GitHub staging acceptance workflow against deployed API.
- Upload release record artifact.
- Verify frontend points at deployed API.

## Phase 23: Race-Day Operations UX

Goal: make the app useful when real race cards are changing during the day.

### 23.1 Next-To-Jump And Live Views

- Add next race, live race, and recently completed race states when provider supports them.
- Show off time, time-to-race, going changes, runner count, non-runner status, and odds freshness.

### 23.2 Race Detail Improvements

- Show provider provenance, last fetched time, field completeness, and model confidence.
- Add runner profile panels for form, trainer/jockey signals, draw, going, and market context.

### 23.3 Operator Workflow

- Add refresh controls that trigger provider import jobs safely.
- Add import status and stale-data warnings beside each race.
- Add watchlists for tracks, runners, jockeys, trainers, and user angles.

## Phase 24: Browser And End-To-End QA

Goal: make visual and workflow testing reliable enough for release blocking.

### 24.1 Agent-Browser Stabilization

- Diagnose local `agent-browser` daemon hangs.
- Add a cleanup/reset helper script for Windows.
- Keep `scripts/visual-smoke-check.py --verbose` as the standard diagnostic mode.

### 24.2 Dedicated E2E Runner

- Add Playwright or another deterministic browser runner for CI-safe UI checks.
- Cover Workspace, Race Centre, Race Card, Evaluation, Monitoring, Bet Journal, Responsible Use, and Admin.
- Run with seeded test DB and no external provider dependency.

### 24.3 Accessibility And Mobile Checks

- Add automated accessibility smoke checks.
- Add mobile viewport assertions for compact race-day use.

## Phase 25: Model Quality And Betting Analytics

Goal: improve the model only after provider depth and governance are reliable.

### 25.1 Champion/Challenger Models

- Add calibrated tree/boosting/ranking candidates.
- Compare candidates against market baseline and approved champion.
- Require race-level holdout and leakage-safe splits.

### 25.2 Odds And Closing-Line Analysis

- Store opening/latest/closing odds when provider terms allow.
- Compare model edge with market movement and settled results.
- Add journal closing-line value and bankroll analytics.

### 25.3 Explainability

- Add runner-level feature contribution summaries.
- Add "why this changed" views between prediction runs.
- Keep uncertainty and limitations visible.

## Phase 26: Product Trust And Legal Readiness

Goal: make the product safe to share outside local development.

### 26.1 Licensing And Display Rules

- Confirm rights to store and display race cards, odds, results, and provider-derived data.
- Add provider attribution and license references where required.
- Add operational checks so unlicensed fields are not displayed.

### 26.2 Responsible Use

- Keep predictions framed as decision support, not betting advice.
- Add configurable responsible-gambling, privacy, and terms links.
- Add safer copy for stale data, thin history, and poor model evaluation.

### 26.3 Public Release Package

- Prepare screenshots, setup docs, demo data, provider setup guide, and release notes.
- Mark what is demo-ready, staging-ready, and production-blocked.

## Recommended First Roadmap03 Step

Start with Phase 19. The platform cannot honestly claim live race-card readiness until authenticated provider imports are proven end to end with stable track matching, DB rows, API endpoints, frontend display, and release evidence.
