# Roadmap03

Roadmap02 is complete through Phase 18. Roadmap03 moves the project from a strong local/staging platform toward live-provider certification, a cost-aware local data broker, optional local AI assistance, account-owned usage, production hosting, and better race-day operation.

## Current Release Check

August 3, 2026 authenticated provider checks found:

- OurHub credentials work and imported 508 live current race-card rows into a fresh SQLite check database
- the redacted provider smoke command passes for OurHub and captures The Racing API plan limits
- OurHub track matching is now course-level, not full race-label-level
- imported OurHub tracks were Clairefontaine, Cork, Naas, Nottingham, Ripon, Vichy, and Windsor
- API health, meetings, race-card filtering, prediction rows, data quality, and provider freshness worked against the OurHub live import
- OurHub field coverage is strong for course, distance, going/surface, horse, jockey, trainer, weight, and draw, but does not currently cover odds or owner
- The Racing API credentials work for `/v1/courses`, but `/v1/racecards` requires Basic Plan and `/v1/results` requires Standard Plan
- user-provided pricing shows The Racing API paid tiers are meaningful monthly costs, so Roadmap03 should avoid making Basic, Standard, or Pro a near-term dependency
- monitoring correctly reports critical drift because the live OurHub card is being compared against the tiny sample historical reference set
- evidence is captured in `docs/LIVE_PROVIDER_CHECK_2026-08-03.md`

August 3, 2026 pre-merge checks found:

- local CI, API, DB, model, release-record, and frontend build paths pass
- internet access works from the machine
- Open-Meteo provider-depth checks work for course enrichment
- OurHub and The Racing API endpoints are reachable
- sample ingestion writes rows into a fresh DB, API track matching works for York, and prediction ranking works
- `agent-browser` can become daemon-blocked on this machine, so browser automation should be stabilized before it becomes the only visual release gate
- Phase 22 account-auth checks now pass with named operator accounts, hashed bearer tokens, role-aware session resolution, account-scoped journal ownership, and an Alembic `operator_accounts` migration
- Phase 23 now has managed-data guardrails for staging/production env validation, MySQL backup/restore rehearsal planning, restore-target safety checks, and CI coverage for those helpers

## Internet And Provider Leads

Checked source leads:

- OurHub Racing API: documents `course-info/{race_date}` and `runner-info/{race_date}` endpoints with `X-API-Key` authentication. Source: https://github.com/TamB10/ourhub-racing-api
- The Racing API: supports racecards/results and requires account credentials. Source: https://api.theracingapi.com/documentation
- BSD Horse Racing API: documents tracks, meetings, races, live races, next-to-jump races, runners, and odds endpoints. Source: https://sports.bzzoiro.com/docs/horseracing/
- FormFav: documents racing data feed, meetings, race form, and predictions resources with a free-tier claim. Source: https://formfav.com/docs
- Open-Meteo historical weather remains useful for no-secret course/weather enrichment checks. Source: https://open-meteo.com/en/docs/historical-weather-api
- Ollama: provides OpenAI-compatible endpoints for connecting apps to local models. Source: https://docs.ollama.com/api/openai-compatibility
- LM Studio: can serve local models through OpenAI-compatible endpoints. Source: https://lmstudio.ai/docs/developer/openai-compat
- llama-cpp-python: offers an OpenAI-compatible local web server for GGUF models. Source: https://llama-cpp-python.readthedocs.io/en/latest/server/
- OpenAI API: remains a hosted, usage-priced API rather than a free local runtime. Source: https://developers.openai.com/api/docs/pricing

## Phase 19: Live Provider Certification

Goal: prove authenticated race-card and result imports from the chosen provider.

### 19.1 Credentials And Secret Handling

- Configure `.env` locally and hosting secrets in staging.
- Current status: local OurHub and The Racing API credentials have been validated without printing secrets.
- Current status: built-in providers use official API hosts by default, even when a stale `HORSE_API_BASE_URL` is present.
- Current status: `scripts/provider-smoke-check.py` redacts credentials and reports endpoint status, row counts, validation issues, field coverage, and acceptance failures.

### 19.2 Live Race-Card Import

- Current status: OurHub imported today's race cards into a throwaway DB and wrote 508 current rows.
- Current status: meetings, track filters, race-card rows, prediction rows, data quality, and ingestion status passed for Windsor.
- Current status: OurHub maps `track`, `distance`, `surface`, `horse`, `jockey`, `trainer`, `draw`, `horse_weight`, and partial `class_rating`; `owner` and `odds` are missing.
- Current status: provider-specific acceptance thresholds allow known OurHub free-tier gaps while still requiring core race-card fields.

### 19.3 Historical Results Import

- Import enough historical rows to make holdout metrics meaningful.
- Current status: The Racing API results endpoint needs Standard Plan before authenticated historical-result import can be tested.
- Current status: The Racing API racecards endpoint needs Basic Plan before authenticated race-card import can be tested from that provider.
- Confirm provider result fields include finishing positions and stable race/runner identifiers.
- Add freshness and minimum-history gates for live-provider acceptance.

### 19.4 Track Alias Normalization

- Add provider course-name alias mapping.
- Treat `York`, `York Racecourse`, and provider-specific spelling as one track.
- Expose alias/provenance warnings in data quality.

## Phase 20: Cost-Aware Local Data Broker And AI Layer

Goal: reduce dependency on expensive paid racing APIs by owning a local data contract and using local AI only where it is safe.

### 20.1 Provider Cost Strategy

- Treat The Racing API paid tiers as optional, not as a Roadmap03 blocker.
- Keep OurHub as the current live race-card source while its limits are understood.
- Verify whether any free-tier The Racing API endpoints can legally and technically support basic import before paying.
- Continue comparing FormFav, BSD Horse Racing, The Odds API, Betfair, Sportradar, and official/licensed data sources.
- Record monthly cost, fields, quotas, usage rights, and production risks before adopting any paid provider.

### 20.2 Local Data Broker API

- Current status: `local_data_broker.py` normalizes manual JSON through the existing provider schema and records provider-result summaries.
- Current status: `data_pipeline.py --cache-raw-payloads` can write raw provider payload envelopes with fetched-at timestamps, source URLs, provider names, hashes, row counts, and licensing notes.
- Current status: `scripts/broker-smoke-check.py` provides a no-network broker fixture and replay check.
- Current status: `/api/v1/broker/status`, `/api/v1/broker/raw-payloads`, and `/api/v1/broker/raw-payload-shape` expose provider cache state and payload structure.
- Next: add app-owned racecards, results, odds, provider plugin status, and import-job controls after the normalized provider entity model is implemented.

### 20.3 Local OpenAI-Compatible AI Layer

- Current status: optional settings exist for `AI_PROVIDER`, `AI_BASE_URL`, `AI_MODEL`, `AI_TIMEOUT_SECONDS`, `AI_JSON_SCHEMA_REQUIRED`, and `AI_API_KEY`.
- Current status: `local_ai.py` supports OpenAI-compatible `/chat/completions` calls for local providers such as Ollama, LM Studio, and llama-cpp-python.
- Current status: AI payload-shape review validates strict JSON and keeps review output separate from provider-supplied facts.
- Current status: local AI is disabled by default and does not block core ingestion.
- Next: add operator-facing review queues for accepted/rejected alias and field-mapping suggestions.

### 20.4 Local AI Smoke Checks

- Current status: `scripts/local-ai-smoke-check.py` provides a no-secret smoke script for `http://127.0.0.1:11434/v1` style local model servers.
- Current status: disabled local AI is tested without network calls, and enabled local AI validates JSON output reliability.
- Current status: docs cover Ollama and LM Studio command examples.
- Keep cloud OpenAI as an optional paid explanation provider, not a required runtime.

### 20.5 Legal And Quality Guardrails

- Do not scrape, store, or redistribute data unless source terms allow it.
- Keep provenance attached to every imported row.
- Require manual approval before adopting new AI-suggested mappings.
- Keep model scoring deterministic; local LLMs may explain or review, but they should not pick winners directly.

See `docs/LOCAL_DATA_BROKER.md` and `docs/LOCAL_AI_STRATEGY.md` for the detailed local broker and local-AI direction.

## Phase 21: Provider Entity Model

Goal: replace flat natural-key upserts with stable provider entities.

### 21.1 Stable Provider IDs

- Current status: normalized sync stores deterministic synthetic provider IDs for courses, meetings, races, entries, horses, jockeys, trainers, and owners when official provider IDs are not available.
- Current status: odds snapshots store provider, odds timestamp, source table, and raw flat-row payload.
- Current status: historical results link to normalized race entries and preserve source/raw payload context.
- Next: replace synthetic IDs with official provider race, runner, horse, jockey, trainer, owner, and odds IDs when the chosen provider contract supplies them.
- Preserve current flat compatibility tables until the frontend/model migrates cleanly.

### 21.2 Normalized Read Models

- Current status: `data_pipeline.py` syncs compatibility rows into `race_meetings`, `races`, `race_entries`, `historical_results`, and `odds_snapshots` after ingestion.
- Current status: `/api/v1/normalized/status` exposes normalized table counts.
- Current status: `/api/v1/normalized/race-entries` exposes an auditable normalized race-entry read model with latest odds and result fields.
- Current status: `scripts/normalized-backfill-check.py` runs a manual sync/read-model check against an existing database.
- Next: switch selected model/API features to normalized read models only after live provider IDs and broader historical data are available.

### 21.3 Provider Comparison Matrix

- Compare OurHub, The Racing API, BSD, FormFav, Betfair, The Odds API, Sportradar, and any official/licensed racing partners.
- Track coverage, cost, authentication, quotas, racecard depth, results depth, odds depth, stable IDs, live latency, and redistribution terms.

## Phase 22: Account Authentication And User Ownership

Goal: replace shared bearer tokens with named users and account-owned data.

### 22.1 Auth Provider

- Current status: `operator_accounts` stores named accounts with SHA-256 hashed bearer tokens.
- Current status: `GET /api/v1/auth/session` resolves account actor, account key/id, roles, auth mode, and environment.
- Current status: viewer, journal, operator, admin, and release-approver roles are normalized and enforced in the API.
- Current status: legacy admin/journal bearer tokens remain a local-development fallback.
- Next: replace account bearer tokens with passwordless/OIDC login when the app moves beyond private operator use.

### 22.2 Account-Owned Bet Journal

- Current status: bet-journal routes use the resolved account key from the access context, so named accounts get isolated journal scopes.
- Current status: `privacy_acknowledged_at` is captured on operator accounts for setup and future privacy controls.
- Next: add user-facing export/delete controls and enforce published privacy-policy links before broad personal-use rollout.

### 22.3 Governance Audit Upgrade

- Current status: `GET /api/v1/admin/accounts` and `POST /api/v1/admin/accounts` expose admin-managed operator accounts.
- Current status: operator-account upserts append `operator_account.upsert` rows to `admin_audit_events`.
- Current status: `scripts/account-smoke-check.py` creates/updates a named account and verifies token lookup without printing token values.
- Next: record auth events, provider config changes, ingestion backfills, release records, and rollback actions as first-class governed events.

## Phase 23: Production Hosting And Managed Data

Goal: deploy a release candidate with managed database, secrets, backups, and release checks.

### 23.1 Managed Database

- Current status: `scripts/managed-db-rehearsal.py` plans or executes MySQL/MariaDB backup and restore rehearsals with redacted output.
- Current status: the rehearsal refuses same-target restores and warns when restore database names do not look isolated.
- Current status: `docs/MANAGED_DATA.md` documents managed DB boundaries, dry-run/execute commands, migration checks, and rollback limits.
- Next: run the rehearsal against a real managed staging source and isolated restore database once cloud database URLs exist.

### 23.2 Secret And Runtime Configuration

- Current status: `scripts/staging-env-check.py` validates deployed env files for managed SQL, HTTPS URLs, allowed hosts, account auth, release tokens, policy links, provider credentials, and placeholder values.
- Current status: staging/production env examples include `ACCOUNT_AUTH_TOKEN` and `RESTORE_DATABASE_URL` checklist fields.
- Current status: Docker staging config allows Phase 22 account auth without requiring legacy shared tokens.
- Next: store real database URLs, provider credentials, policy links, and account tokens only in the chosen host's secret store.

### 23.3 Staging Acceptance

- Current status: CI compiles the deployment helper scripts and dry-runs staging env plus managed DB rehearsal checks.
- Current status: `.github/workflows/staging-acceptance.yml` prefers `ACCOUNT_AUTH_TOKEN` with legacy `API_AUTH_TOKEN` fallback.
- Next: run the manual staging acceptance workflow against a deployed API URL and upload the release record artifact.

## Phase 24: Race-Day Operations UX

Goal: make the app useful when real race cards are changing during the day.

### 24.1 Next-To-Jump And Live Views

- Current status: `/api/v1/race-day` returns race-level operational status from the scored current race card.
- Current status: the endpoint identifies next/live/race-day/upcoming/complete/stale states, exposes off time and minutes-to-post when provider fields are present, and keeps provider/data-age provenance attached to each race.
- Current status: the Race Centre now has a race-day board with next visible race, post window, provider, data age, top runner, and status counts.
- Next: add true provider next-to-jump, non-runner, going-change, and odds-freshness fields once the selected provider supplies them.

### 24.2 Race Detail Improvements

- Current status: selected Race Centre detail shows status label, off-time window, market favorite, provider, last import, and data age beside race context.
- Current status: race list items show compact status pills so stale or upcoming cards are visible before selection.
- Next: add runner profile panels for form, trainer/jockey signals, draw, going, market context, and field completeness.

### 24.3 Operator Workflow

- Current status: race-level stale-data warnings are visible through `/api/v1/race-day` and the Race Centre status pills.
- Next: add refresh controls that trigger provider import jobs safely.
- Add watchlists for tracks, runners, jockeys, trainers, and user angles.

## Phase 25: Browser And End-To-End QA

Goal: make visual and workflow testing reliable enough for release blocking.

### 25.1 Agent-Browser Stabilization

- Current status: `scripts/agent-browser-reset.ps1` and `scripts/agent-browser-reset.sh` close stuck sessions, clean up helper processes, and run `agent-browser doctor --fix` with a timeout.
- Current status: `scripts/visual-smoke-check.py --reset-browser` can reset the daemon before opening the frontend.
- Keep `scripts/visual-smoke-check.py --reset-browser --verbose` as the standard diagnostic mode when the local preview is unstable.

### 25.2 Dedicated E2E Runner

- Current status: the agent-browser visual gate now covers Workspace, Race Centre, Race-Day Board, Race Card, Evaluation, Bet Journal, Responsible Use, and optional Admin.
- Current status: CI syntax-checks the agent-browser reset shell helper.
- Next: add Playwright or another deterministic browser runner for CI-safe UI checks.
- Cover Monitoring in the browser flow after its DOM anchors are made stable enough for automation.
- Run with seeded test DB and no external provider dependency.

### 25.3 Accessibility And Mobile Checks

- Current status: the visual smoke check still asserts the app renders at a compact mobile viewport.
- Next: add automated accessibility smoke checks.
- Next: add mobile viewport assertions for compact Race Centre and race-day board layout.

## Phase 26: Model Quality And Betting Analytics

Goal: improve the model only after provider depth and governance are reliable.

### 26.1 Champion/Challenger Models

- Add calibrated tree/boosting/ranking candidates.
- Compare candidates against market baseline and approved champion.
- Require race-level holdout and leakage-safe splits.

### 26.2 Odds And Closing-Line Analysis

- Store opening/latest/closing odds when provider terms allow.
- Compare model edge with market movement and settled results.
- Add journal closing-line value and bankroll analytics.

### 26.3 Explainability

- Add runner-level feature contribution summaries.
- Add "why this changed" views between prediction runs.
- Keep uncertainty and limitations visible.

## Phase 27: Product Trust And Legal Readiness

Goal: make the product safe to share outside local development.

### 27.1 Licensing And Display Rules

- Confirm rights to store and display race cards, odds, results, and provider-derived data.
- Add provider attribution and license references where required.
- Add operational checks so unlicensed fields are not displayed.

### 27.2 Responsible Use

- Keep predictions framed as decision support, not betting advice.
- Add configurable responsible-gambling, privacy, and terms links.
- Add safer copy for stale data, thin history, and poor model evaluation.

### 27.3 Public Release Package

- Prepare screenshots, setup docs, demo data, provider setup guide, and release notes.
- Mark what is demo-ready, staging-ready, and production-blocked.

## Recommended Next Roadmap03 Step

Phase 25 now has agent-browser reset helpers and stronger visual smoke coverage for the Race Centre and Race-Day Board. Move next into Phase 26: Model Quality And Betting Analytics, while keeping a future task open for a CI-safe Playwright-style runner.
