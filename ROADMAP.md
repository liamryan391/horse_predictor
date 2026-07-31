# Horse Predictor Roadmap

This roadmap turns the current local prototype into a reliable racing intelligence platform. The order is intentional: improve data trust first, then model quality, then product and deployment polish.

## Phase 1: Stabilise The Data Foundation

- Add provider-specific schema validation before database writes.
- Store ingestion run metadata, errors, row counts, and provider response timings.
- Keep raw API payload samples for debugging and provider mapping changes.
- Add deduplication rules for races, runners, horses, jockeys, trainers, and tracks.
- Build repeatable seed scripts for local SQLite and production MySQL.

## Phase 2: Improve Prediction Quality

- Add train/test race-level splits so horses from the same race stay in the same evaluation set.
- Track model metrics including log loss, calibration, top-pick hit rate, and place-rate accuracy.
- Add backtesting against historical odds, including profit simulation and drawdown.
- Add richer features: going, distance, class, official rating, recent form, trainer strike rate, jockey strike rate, and market movement.
- Persist model versions and metrics in SQL instead of training only at request time.

## Phase 3: Productise The Race Lab

- Add race detail pages with runner comparisons, feature explanations, and confidence bands.
- Add filters for date, track, race type, going, distance, and market range.
- Add saved shortlists for races or horses to watch.
- Add alerts for major odds movement or model confidence changes.
- Keep the Streamlit app as an internal diagnostic tool while React becomes the primary interface.

## Phase 4: Production Backend And Operations

- Add authentication before exposing the API beyond localhost.
- Move scheduled ingestion into APScheduler, Celery, hosted cron, or another managed worker.
- Add API rate-limit handling, retries with backoff, and provider outage reporting.
- Add structured logging and health checks for database, ingestion, and model readiness.
- Add automated tests for ingestion mapping, storage writes, prediction responses, and frontend build.

## Phase 5: Deployment

- Provision managed MySQL for production data.
- Deploy the FastAPI backend with environment-managed secrets.
- Deploy the React frontend as a static site connected to the backend API.
- Add separate development, staging, and production environment settings.
- Document backup, restore, and data retention procedures.

## Responsible Use

- Present predictions as decision support, not guaranteed betting advice.
- Show uncertainty and data freshness wherever predictions are displayed.
- Avoid hiding model limitations, missing data, or provider outages.
- Add bankroll and risk warnings before any public release.
