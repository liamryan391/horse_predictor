# Prediction Modeling

Phase 4 adds an explicit evaluation layer around the current logistic-regression model. The model is still intentionally simple, but the code now separates result-only targets from pre-race features and reports chronological holdout metrics.

## Leakage Guard

Result-only fields are blocked from model inputs:

- `finishing_position`
- `is_winner`

`race_date` is also excluded from direct predictors. Calendar signals are exposed through engineered fields such as `race_month` and `race_day_of_week`.

## Pre-Race Features

The shared model module builds these race-context features for both training and scoring:

- `country`
- `distance_bucket`
- `going_category`
- `race_type`
- `implied_probability`
- `field_size`
- `odds_rank`
- `relative_speed_rating`
- `relative_class_rating`

These features are based on race-card information that should be knowable before the race.

Course metadata, distance buckets, going categories, and race type are derived by `race_enrichment.py`. Coverage is visible through `/api/v1/data-quality` and can be enforced during production readiness checks with `--require-enriched-data`.

## Holdout Evaluation

`evaluate_model` sorts races chronologically, trains on earlier races, and validates on the latest holdout races. It reports:

- runner log loss and Brier score
- market baseline log loss and Brier score
- calibration mean absolute error
- model and market top-pick win rates
- mean winner rank
- fixed-stake value-bet profit and ROI

The sample dataset is intentionally tiny, so its metrics are only a wiring smoke test. Provider history will make the evaluation meaningful.

## Artifact Serving

`POST /api/v1/admin/model/evaluation` trains the current candidate and writes a serialized `ModelResult` artifact with feature-schema metadata. `POST /api/v1/admin/model/{model_version_id}/approve` verifies the artifact checksum and feature-schema hash before making it the approved serving model.

Prediction routes load the latest approved artifact when one exists. Local development can fall back to in-memory training, while staging and production should set `REQUIRE_APPROVED_MODEL_ARTIFACT=true`.

## API

Model evaluation is returned by:

```text
GET /api/model
GET /api/model/evaluation
```

The React Race Lab and Streamlit app both read the shared model module so leakage checks, artifact save/load checks, and evaluation logic stay consistent.
