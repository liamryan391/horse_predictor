# Horse Racing Software + App

This project includes a Streamlit app that can:

- Read **past race data** and **past bet history data** from CSV.
- Analyze important details such as jockey, owner, trainer, odds, and horse stats.
- Train a machine-learning model from previous data.
- Score current races and rank horses by estimated win probability.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run horse_racing_app.py
```

Then upload:

- `sample_historical_data.csv` (or your own historical file)
- `sample_current_races.csv` (or your own upcoming race card)

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
- `finishing_position` (historical data should contain true result; current race card can use `0` as placeholder)

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
- Improve model quality by adding more historical races, sectional times, and richer features.
