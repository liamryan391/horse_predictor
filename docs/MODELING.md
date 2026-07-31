# Prediction Modeling

Phase 4 adds an explicit evaluation layer around the current logistic-regression model. The model is still intentionally simple, but the code now separates result-only targets from pre-race features and reports chronological holdout metrics.

## Leakage Guard

Result-only fields are blocked from model inputs:

- `finishing_position`
- `is_winner`

`race_date` is also excluded from direct predictors. Calendar signals are exposed through engineered fields such as `race_month` and `race_day_of_week`.

## Pre-Race Features

The shared model module builds these race-context features for both training and scoring:

- `implied_probability`
- `field_size`
- `odds_rank`
- `relative_speed_rating`
- `relative_class_rating`

These features are based on race-card information that should be knowable before the race.

## Holdout Evaluation

`evaluate_model` sorts races chronologically, trains on earlier races, and validates on the latest holdout races. It reports:

- runner log loss and Brier score
- market baseline log loss and Brier score
- calibration mean absolute error
- model and market top-pick win rates
- mean winner rank
- fixed-stake value-bet profit and ROI

The sample dataset is intentionally tiny, so its metrics are only a wiring smoke test. Provider history will make the evaluation meaningful.

## API

Model evaluation is returned by:

```text
GET /api/model
GET /api/model/evaluation
```

The React Race Lab and Streamlit app both read the shared model module so leakage checks and evaluation logic stay consistent.
