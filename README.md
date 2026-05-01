# Horse Racing Software + App

This project now has **two parts**:

1. **Data Pipeline API Ingestion** (`data_pipeline.py`) to fetch horse info and build datasets.
2. **Prediction App** (`horse_racing_app.py`) to train on past races and score current races.

## Should SQL be used?

Yes — SQL is a good fit for this use case.

- It keeps race/bet/jockey/owner tables structured and queryable.
- You can track ingestion timestamps and history cleanly.
- It scales better than only CSV files when data grows.

This project includes SQLite support in `data_pipeline.py` and creates:

- `races_historical` table
- `races_current` table

You can later switch to Postgres/MySQL with the same table design.

## API ingestion setup

Set API credentials (example):

```bash
export HORSE_API_BASE_URL="https://api.your-provider.com/v1"
export HORSE_API_KEY="your_api_key"
```

Run ingestion:

```bash
python data_pipeline.py --days-ahead 7
```

This will:

- Call `/historical-races`
- Call `/current-races?days_ahead=7`
- Create `sample_historical_data.csv`
- Create `sample_current_races.csv`
- Load both datasets into `horse_racing.db`

## Prediction app setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run horse_racing_app.py
```

Then upload the generated CSVs (or your own files).

## Required CSV columns

- `race_date`
- `track`
- `distance`
- `surface`
- `horse`
- `jockey`
- `owner`
- `trainer`
- `odds`
- `finishing_position`

## Optional columns

- `horse_age`
- `horse_weight`
- `draw`
- `speed_rating`
- `class_rating`
- `days_since_last_run`
- `past_bets_count`
- `past_bets_profit`
- `weather`

## Notes

- This is decision-support software, not guaranteed betting advice.
- For production: add API retries, schema validation, and automated model evaluation.
