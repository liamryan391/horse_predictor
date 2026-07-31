# Future Roadmap

This document captures longer-term development ideas for Horse Predictor after `ROADMAP02.md`. It is intentionally broader than the next build roadmap and should be reviewed again after Phase 12 through Phase 18 are complete.

## Future Development Principles

- Keep the app as decision-support software, not automated betting advice.
- Verify data-provider licensing before storing, redistributing, or displaying provider data.
- Prefer auditable model and prediction workflows over opaque shortcuts.
- Prioritize user trust: freshness, uncertainty, holdout metrics, and responsible-use copy should stay visible.
- Treat every paid API, exchange API, and odds feed as a product/legal decision before it becomes an engineering task.

## Data And Provider Strategy

Longer-term model quality will depend on richer racing context than the current sample-compatible flat schema can provide.

Future ideas:

- Build a provider comparison matrix for The Racing API, OurHub Racing API, Odds API, Betfair, and any official/licensed racing data partners.
- Add a provider abstraction for racecards, results, runner history, odds snapshots, ratings, course metadata, and market movements.
- Store provider provenance on every imported row so model outputs can be traced back to source and timestamp.
- Add regional provider support only where the app has enough historical depth to evaluate fairly.
- Add provider cost, quota, terms, and redistribution notes to operational docs before production use.

## Market And Odds Intelligence

Odds are useful as both model inputs and product context, but live market integrations need careful handling.

Future ideas:

- Track opening price, latest price, closing price, and market movement when provider terms permit.
- Compare model probability with market-implied probability to identify possible value, while keeping uncertainty prominent.
- Add market movement charts on race detail pages.
- Add closing-line comparison to the bet journal after results settle.
- Investigate Betfair Exchange data only after licensing, account verification, and allowed-use constraints are confirmed.

Do not build automated order placement or trading flows unless the product direction, licensing, safeguards, and user controls are explicitly approved.

## Advanced Modelling

Once artifact-backed serving is stable, the modelling roadmap can move beyond the current logistic baseline.

Future ideas:

- Add candidate model families such as gradient boosting, calibrated classifiers, and ranking models.
- Compare candidates against the currently approved champion before promotion.
- Add calibration plots, Brier score, expected calibration error, and rank-based metrics.
- Add race-level cross-validation so leakage between runners in the same race is controlled.
- Add feature stores or generated feature snapshots if provider data grows beyond simple tables.
- Add scheduled retraining with promotion rules rather than automatic replacement.

## Explainability And Trust

The race centre should explain why a runner is ranked highly without pretending the model is certain.

Future ideas:

- Add model-feature contribution summaries for each runner.
- Add SHAP-backed explanations after artifact-backed serving and feature-schema versioning are reliable.
- Add "why this changed" views when prediction runs differ after a data refresh.
- Add model-vs-market-vs-result comparison panels for completed races.
- Add plain-language caveats when data is stale, incomplete, or outside the model's strongest training region.

## Product Experience

The professional dashboard can grow into a daily racing workspace.

Future ideas:

- Add a race centre with race cards, runner comparisons, model ranks, market odds, notes, and settlement status.
- Add saved filters for country, course, race type, going, distance, class, and value threshold.
- Add watchlists for horses, trainers, jockeys, courses, and user-defined angles.
- Add mobile-first race navigation for users checking cards near race time.
- Add a progressive web app shell and notification preferences after authentication exists.

## Bet Journal And User Accounts

The journal becomes much more useful when it is server-side, private, and linked to settled results.

Future ideas:

- Add accounts and roles before storing personal bet history.
- Add server-side bet journal tables with export and delete controls.
- Link journal rows to prediction runs, model versions, market odds, and final results.
- Add user-level performance analytics, bankroll exposure, drawdown, and closing-line comparison.
- Add responsible-use controls such as stake reminders, cool-off copy, and configurable risk limits.

## Operations And Monitoring

Production confidence should come from repeatable checks, not memory.

Future ideas:

- Add OpenTelemetry traces and metrics for API requests, ingestion jobs, scoring, and model approval actions.
- Add data drift and prediction drift reports using reference and current windows.
- Add alerts for stale provider data, missing approved models, failed prediction-run persistence, failed migrations, and model-quality regression.
- Add scheduled smoke tests for API, frontend, database, ingestion, model registry, and prediction-run history.
- Keep `agent-browser` visual verification as a future gate once local preview access is stable.

## Admin And Governance

Administrative actions should become visible, reversible where possible, and auditable.

Future ideas:

- Add an admin console for ingestion runs, provider freshness, model registry, approvals, prediction runs, and launch gates.
- Add audit logs for model approval, rollback, ingestion backfill, provider configuration changes, and admin actions.
- Add role-based access for viewers, journal users, operators, and admins.
- Add release records that include commit SHA, migration head, approved model id, prediction-run id, provider freshness, and rollback target.

## Possible Research Tracks

These are useful exploratory tracks, but none should displace the near-term model-serving work.

- Course-specific and going-specific model slices.
- Race-type-specific models for flat, jumps, handicaps, and group races.
- Runner form embeddings based on previous race sequence.
- Trainer/jockey/course interaction features.
- Weather and soil-moisture effects by course and distance.
- Market disagreement alerts where model, public odds, and ratings point in different directions.
- Natural-language race summaries generated from structured predictions and caveats.

## Future Source Leads

These leads are worth revisiting when planning later phases:

- The Racing API: https://www.theracingapi.com/
- The Racing API data coverage: https://www.theracingapi.com/data-coverage
- OurHub Racing API: https://github.com/TamB10/ourhub-racing-api
- Betfair Exchange API: https://developer.betfair.com/exchange-api/
- Betfair API licensing note: https://support.developer.betfair.com/hc/en-us/articles/360002464152-Which-API-Licence-do-I-require-
- Open-Meteo historical weather API: https://open-meteo.com/en/docs/historical-weather-api
- BHA ratings database: https://www.britishhorseracing.com/regulation/official-ratings/ratings-database/
- Odds API racing reference: https://api.odds-api.net/v1/reference
- Racing Post racecard UX update: https://www.racingpost.com/news/find-out-what-is-new-on-the-updated-racing-post-racecards-agNRD5B1qzui/
- MLflow model registry workflow: https://www.mlflow.org/docs/latest/ml/model-registry/workflow/
- Evidently data drift docs: https://docs.evidentlyai.com/metrics/preset_data_drift
- SHAP explainability docs: https://shap.readthedocs.io/en/stable/api.html
- OpenTelemetry FastAPI instrumentation: https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html

## Best Future Bets

The strongest future investments are:

1. Better data depth with clear licensing.
2. Reproducible model artifacts and champion/rollback serving.
3. Race detail UX with transparent runner explanations.
4. Server-side journal linked to prediction runs and results.
5. Monitoring for data freshness, drift, and model-quality regression.

These ideas should stay behind `ROADMAP02.md` until Phase 12 through Phase 18 have made the platform stable enough to support them.
