# Roadmap02

This is the next build roadmap after the original `ROADMAP.md` reached Phase 11. It focuses on the next practical product and platform steps for Horse Predictor. Broader future ideas are captured in `FUTURE_ROADMAP.md`.

## Phase Boundary Check

Phase 10 and Phase 11 are separate commits:

- Phase 10: `765c3f8` - `Add phase 10 model operations`
- Phase 11: `bee37df` - `Add phase 11 prediction operations`

Phase 11 followed Phase 10 quickly in the same working thread because the original roadmap stopped at Phase 10. Phase 11 was created from the remaining Phase 10 lifecycle items: prediction-run persistence, prediction-run API history, and launch evidence gates.

## Source Leads Reviewed

These sources shaped the next roadmap:

- The Racing API: https://www.theracingapi.com/
- The Racing API documentation: https://api.theracingapi.com/
- The Racing API data coverage: https://www.theracingapi.com/data-coverage
- The Racing API ratings notes: https://www.theracingapi.com/tools/ratings
- OurHub Racing API: https://github.com/TamB10/ourhub-racing-api
- Betfair Exchange API: https://developer.betfair.com/exchange-api/
- Betfair API licensing note: https://support.developer.betfair.com/hc/en-us/articles/360002464152-Which-API-Licence-do-I-require-
- Open-Meteo weather APIs: https://open-meteo.com/
- BHA ratings database: https://www.britishhorseracing.com/regulation/official-ratings/ratings-database/
- Odds API racing reference: https://api.odds-api.net/v1/reference
- Racing Post racecard UX update: https://www.racingpost.com/news/find-out-what-is-new-on-the-updated-racing-post-racecards-agNRD5B1qzui/
- MLflow model registry workflow: https://www.mlflow.org/docs/latest/ml/model-registry/workflow/
- Evidently data drift docs: https://docs.evidentlyai.com/metrics/preset_data_drift
- SHAP explainability docs: https://shap.readthedocs.io/en/stable/api.html
- OpenTelemetry FastAPI instrumentation: https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html

## Phase 12: Artifact-Backed Model Serving

Goal: stop retraining inside each scoring path and serve predictions from an approved, reproducible model artifact.

### 12.1 Model Artifact Persistence

- Serialize trained sklearn pipelines with feature schema metadata.
- Store artifact URI, feature schema hash, training data window, and code commit SHA.
- Load approved artifacts for scoring instead of retraining in request handlers.
- Add artifact integrity checks before approval.

Current inspection: `POST /api/v1/admin/model/evaluation` now writes a serialized model artifact under `MODEL_ARTIFACT_DIR`, stores artifact URI/checksum/schema/commit metadata in `model_versions`, and keeps artifact files out of git through `model_artifacts/`.

### 12.2 Champion Model Runtime

- Add a `champion` model lookup path that resolves the current approved model.
- Keep the existing in-memory retraining path only as a local fallback.
- Add a startup readiness failure when production has no approved artifact.
- Add rollback support to re-point serving to a prior approved artifact.

Current inspection: prediction scoring now uses the latest approved artifact when available. `REQUIRE_APPROVED_MODEL_ARTIFACT=true` makes readiness and prediction serving fail without a loadable approved artifact. Rollback is supported by approving a prior artifact-backed model version.

### 12.3 Validation

- Test artifact save/load roundtrips.
- Test prediction determinism before and after artifact reload.
- Test rejection of feature-schema mismatches.

Current inspection: tests now cover artifact save/load determinism, registry artifact metadata, artifact-backed model approval, prediction-run linkage to the approved artifact, and Alembic upgrade/downgrade coverage for artifact metadata columns.

## Phase 13: Data Enrichment And Provider Depth

Goal: improve model inputs with provider-grade racing context rather than relying mainly on sample flat tables.

### 13.1 Provider Selection Spike

- Compare The Racing API, OurHub, Odds API, and any licensed provider offers.
- Score each provider for cost, historical depth, racecard fields, odds history, rate limits, redistribution rights, and support.
- Choose one primary provider and one fallback provider.

Current inspection: The project now has a provider-depth smoke script and docs for live Open-Meteo weather checks, while real racing provider selection remains a licensing/commercial decision. The Racing API and OurHub adapters remain available behind credentials.

### 13.2 Race Context Enrichment

- Add official ratings import where licensing permits, starting with the BHA ratings export for British runners.
- Add course metadata, country, race type, distance buckets, and track condition fields.
- Add going/weather enrichment using Open-Meteo forecast and historical data by course coordinates.
- Add odds movement snapshots when provider terms permit display and storage.

Current inspection: `race_enrichment.py` now derives course country/coordinates, distance buckets, going categories, and race type for training, scoring, API prediction rows, and provider adapter outputs. Open-Meteo historical daily weather lookup is available through `scripts/provider-depth-check.py`.

### 13.3 Data Quality Gates

- Add provider freshness checks per table, not only globally.
- Track missing-value rates by field and provider.
- Reject model training when key features have drifted or dropped below coverage thresholds.

Current inspection: `/api/v1/data-quality` reports row-quality issues, enrichment coverage, and latest provider freshness per provider/table. The React Evaluation view displays this signal, and `scripts/production-readiness-check.py --require-enriched-data` can enforce core enrichment coverage.

## Phase 14: Race Centre UX

Goal: turn the React workspace into a race-by-race operating centre.

### 14.1 Race Detail View

- Add a dedicated race page with runners, prediction ranks, market odds, model odds, value edge, form context, and notes.
- Add runner comparison panels inside a race rather than only global top two.
- Add mobile-friendly sorting by odds, rank, draw, speed rating, and model value.

Current inspection: the React workspace now includes a Race Centre tab that groups predictions by race, keeps a selected race list/detail layout, shows race context, and supports runner sorting by model rank, value edge, market odds, draw, and speed rating.

### 14.2 Explanation Layer

- Add per-runner explanation cards showing the strongest positive and negative factors.
- Start with transparent model-feature contributions from the existing logistic model.
- Add SHAP-backed explanations after artifact-backed serving is stable.

Current inspection: selected-race runner cards and the runner table now show bounded deterministic signal labels from available model inputs: model rank, value edge, market favourite, speed/class context, draw, going, and race type. True coefficient/SHAP contribution views remain future work.

### 14.3 Responsible Product Copy

- Keep uncertainty, holdout metrics, and data freshness visible on the race page.
- Avoid any "guaranteed winner" or automatic-betting language.

Current inspection: the Race Centre remains inside the existing workspace, so holdout, freshness, and data-quality metrics stay visible above the race view. The copy stays in decision-support language.

## Phase 15: Server-Side Bet Journal

Goal: make the journal useful across devices and auditable for model feedback without turning it into betting advice.

### 15.1 API And Database

- Move bet journal rows from browser local storage into server-side tables.
- Add user/account ownership before storing personal bet history.
- Add settlement status, result linkage, closing odds, and notes.

Current inspection: `/api/v1/bet-journal` now supports list, create, patch, and delete operations backed by `user_bets`. The table now supports operator-local ownership, manual horse/track/date context, nullable race-entry linkage, optional prediction-run/model links, closing odds, settlement status, notes, and server-computed profit/loss. Phase 17 adds bearer-token journal scoping; real user authentication and personal account ownership remain future work.

### 15.2 Analytics

- Compare recorded bets with model probability, market odds, and closing line where available.
- Separate user performance, model performance, and market baseline.
- Add drawdown and bankroll exposure views.

Current inspection: the React Bet Journal reads server rows, records placed odds and optional closing odds, and keeps journal profit/ROI metrics separate from model-evaluation metrics. Model-vs-market-vs-user attribution, drawdown, and bankroll exposure remain future analytics work.

### 15.3 Safeguards

- Add stake warnings and responsible-use reminders.
- Add export/delete support for privacy and account control.

Current inspection: the UI keeps server-backed journal records behind decision-support language, supports row deletion through the API, and falls back to browser storage if the local API is unavailable. Export controls, account-level privacy tools, and stake warnings remain future work.

## Phase 16: Monitoring And Drift

Goal: know when data, predictions, or model quality are changing.

### 16.1 Drift Reports

- Add reference vs current feature drift checks.
- Add prediction distribution drift checks.
- Add alerts when drift affects core fields such as odds, field size, ratings, going, or market-implied probability.

Current inspection: `/api/v1/monitoring` now compares historical reference features with current race-card features, and compares reference/current prediction distributions when model serving is available. Drift rows cover numeric and categorical fields, use configurable warning/critical thresholds, and produce operator alerts for warning or critical drift.

### 16.2 Observability

- Add OpenTelemetry FastAPI instrumentation.
- Emit request traces, slow endpoint spans, ingestion job spans, and model-scoring spans.
- Add dashboard-ready metrics for ingestion freshness, prediction-run volume, API errors, and model evaluation status.

Current inspection: FastAPI request middleware now records in-process request counts, status buckets, top paths, average latency, slow-request counts, and recent request samples. Optional OpenTelemetry FastAPI instrumentation is available through `OTEL_ENABLED`; if the package is unavailable the API logs the condition and continues. JSON logging includes request status and slow-request flags.

### 16.3 Operator Alerts

- Add alert routes or webhook hooks for failed ingestion, stale data, missing approved model, and failed prediction-run capture.

Current inspection: `/api/v1/monitoring` returns active alerts for drift, stale or missing freshness, provider failures, empty race tables, model-serving blocks, missing approved model evidence, missing prediction-run snapshots, high API error rate, and slow requests. The React workspace now includes a Monitoring tab with alert, drift, and API traffic tables, and `scripts/production-readiness-check.py --require-monitoring --require-no-critical-alerts` can enforce the monitoring gate.

## Phase 17: Admin Console And Governance

Goal: make operational actions visible and controlled.

### 17.1 Admin UI

- Add an authenticated admin page for ingestion runs, model snapshots, approvals, prediction runs, and readiness checks.
- Add approve/supersede model actions with confirmation.
- Add audit log rows for admin actions.

Current inspection: the React app now includes an Admin tab with stored bearer-token and actor controls, governance/readiness metrics, model snapshot capture, model approval, model supersede, approved-model prediction-run capture, sample seeding, ingestion visibility, model registry rows, prediction-run rows, and audit history. Governed actions use confirmation prompts.

### 17.2 Auth And Roles

- Add user login before server-side bet journal or admin UI.
- Separate read-only users, journal users, and admins.
- Keep admin endpoints bearer-token protected until proper auth is in place.

Current inspection: FastAPI now has an access context with `reader`, `journal`, and `admin` roles. `API_AUTH_TOKEN` grants admin plus journal access, `JOURNAL_AUTH_TOKEN` grants journal access, and local development falls back to `local-dev` only when no tokens are configured. Staging and production require non-placeholder admin and journal tokens, but full named-user login remains a later authentication phase.

### 17.3 Audit And Release Checks

- Add audit storage for admin actions.
- Add admin governance read endpoints for release verification.
- Add optional production-readiness checks for admin session, governance, and audit-log routes.

Current inspection: `admin_audit_events` stores actor, roles, action, resource, request id, status, detail, payload, and timestamp. `/api/v1/admin/session`, `/api/v1/admin/governance`, and `/api/v1/admin/audit-log` expose the console and release-check contract, and `scripts/production-readiness-check.py --require-admin-governance` can validate the authenticated governance surface.

## Phase 18: CI, Release, And Visual Gates

Goal: turn the current manual validation list into repeatable automation.

### 18.1 GitHub Actions

- Add backend tests, frontend tests, frontend build, Python compile, Docker config, and migration checks to CI.
- Add Dependabot review workflow notes.
- Block merge when tests fail.

Current inspection: `.github/workflows/ci.yml` now runs backend compile/migrations/tests, frontend install/tests/build, shell syntax, Docker Compose config, and diff-hygiene checks on PRs into `development`/`main` and direct pushes to those protected branches. `.github/dependabot.yml` already covers pip, npm, and GitHub Actions updates, and the PR template now asks for CI, visual, release-record, and token-handling checks.

### 18.2 Browser Verification

- Add `agent-browser` visual smoke checks once local preview access is stable.
- Cover workspace, race card, evaluation, journal, responsible-use page, and mobile viewport.

Current inspection: `scripts/visual-smoke-check.py` now uses `agent-browser` to verify the frontend has content, has no framework error overlay, and can navigate Workspace, Bet Journal, Responsible Use, and Admin when an admin token is supplied. The testing docs now treat this as the visual gate for UI or staging acceptance work.

### 18.3 Release Checklist

- Add a single release command or workflow for staging acceptance.
- Record commit SHA, migration head, approved model id, prediction-run id, and data freshness at release time.

Current inspection: `.github/workflows/staging-acceptance.yml` can run release-grade readiness gates and upload a JSON release record. `scripts/staging-release.ps1` and `scripts/staging-release.sh` can also run acceptance gates and write release evidence when `RUN_ACCEPTANCE_CHECKS`, `API_BASE_URL`, and `RELEASE_RECORD_PATH` are set. `scripts/release-record.py` captures commit, Alembic state, readiness, freshness, monitoring, approved model artifact, latest prediction run, ingestion rows, and optional admin governance audit evidence.

## Recommended Next Phase

Roadmap02 Phases 12 through 18 are now implemented. Choose the next build from `FUTURE_ROADMAP.md`.

Recommended next direction: account-backed authentication and user ownership.

Reason: Phase 18 gives the project repeatable CI and release gates. The next platform risk is replacing shared bearer tokens and operator-local journal ownership with named users, durable roles, account-scoped bet history, and stronger admin governance.
