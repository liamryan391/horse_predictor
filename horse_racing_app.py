from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from racing_storage import (
    RACE_COLUMNS,
    TABLES,
    ingestion_status,
    read_races,
    seed_database_from_samples,
    table_counts,
)
from settings import get_settings

SETTINGS = get_settings()
DATABASE_URL = SETTINGS.database_url

REQUIRED_HISTORICAL_COLUMNS = [
    "race_date",
    "track",
    "distance",
    "surface",
    "horse",
    "jockey",
    "owner",
    "trainer",
    "odds",
    "finishing_position",
]

REQUIRED_CURRENT_COLUMNS = [col for col in REQUIRED_HISTORICAL_COLUMNS if col != "finishing_position"]

OPTIONAL_COLUMNS = [
    "horse_age",
    "horse_weight",
    "draw",
    "speed_rating",
    "class_rating",
    "days_since_last_run",
    "past_bets_count",
    "past_bets_profit",
    "weather",
]


@dataclass
class ModelResult:
    model: Pipeline
    feature_columns: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    training_rows: int
    winner_rate: float


def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
    return [col for col in required_columns if col not in df.columns]


def coerce_race_frame(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    for column in RACE_COLUMNS:
        if column not in work_df.columns:
            work_df[column] = None

    work_df["race_date"] = pd.to_datetime(work_df["race_date"], errors="coerce")
    numeric_columns = [
        "distance",
        "odds",
        "finishing_position",
        "horse_age",
        "horse_weight",
        "draw",
        "speed_rating",
        "class_rating",
        "days_since_last_run",
        "past_bets_count",
        "past_bets_profit",
    ]
    for column in numeric_columns:
        work_df[column] = pd.to_numeric(work_df[column], errors="coerce")
    return work_df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    work_df = coerce_race_frame(df)
    work_df = work_df.dropna(subset=["finishing_position"]).copy()

    work_df["race_month"] = work_df["race_date"].dt.month
    work_df["race_day_of_week"] = work_df["race_date"].dt.dayofweek
    work_df["is_winner"] = (work_df["finishing_position"] == 1).astype(int)
    work_df["implied_probability"] = 1 / work_df["odds"].replace(0, np.nan)

    columns = RACE_COLUMNS + ["race_month", "race_day_of_week", "implied_probability", "is_winner"]
    return work_df[columns]


def train_model(df: pd.DataFrame) -> ModelResult:
    if df.empty:
        raise ValueError("No historical rows are available for training.")
    if df["is_winner"].nunique() < 2:
        raise ValueError("Historical data needs at least one winner and one non-winner.")

    excluded = {"is_winner", "finishing_position", "race_date"}
    features = [col for col in df.columns if col not in excluded]
    numeric_features = [col for col in features if pd.api.types.is_numeric_dtype(df[col])]
    categorical_features = [col for col in features if col not in numeric_features]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    X = df[features]
    y = df["is_winner"]
    model.fit(X, y)

    return ModelResult(
        model=model,
        feature_columns=list(X.columns),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        training_rows=len(df),
        winner_rate=float(y.mean()),
    )


def add_scoring_features(races_df: pd.DataFrame) -> pd.DataFrame:
    score_df = coerce_race_frame(races_df)
    score_df["race_month"] = score_df["race_date"].dt.month
    score_df["race_day_of_week"] = score_df["race_date"].dt.dayofweek
    score_df["implied_probability"] = 1 / score_df["odds"].replace(0, np.nan)
    return score_df


def score_current_races(model_result: ModelResult, races_df: pd.DataFrame) -> pd.DataFrame:
    score_df = add_scoring_features(races_df)
    X = score_df.reindex(columns=model_result.feature_columns)
    score_df["win_probability"] = model_result.model.predict_proba(X)[:, 1]
    score_df["model_odds"] = np.where(score_df["win_probability"] > 0, 1 / score_df["win_probability"], np.nan)
    score_df["value_edge"] = score_df["win_probability"] - score_df["implied_probability"]

    race_group = ["race_date", "track", "distance"]
    score_df["suggested_rank"] = score_df.groupby(race_group, dropna=False)["win_probability"].rank(
        ascending=False,
        method="min",
    )
    return score_df.sort_values(race_group + ["suggested_rank", "win_probability"], ascending=[True, True, True, True, False])


def summarize_entities(history_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    summaries = {}
    for entity in ["jockey", "trainer", "owner"]:
        summary = (
            history_df.groupby(entity, dropna=False)
            .agg(runs=("horse", "count"), wins=("is_winner", "sum"), avg_odds=("odds", "mean"))
            .assign(win_rate=lambda x: x["wins"] / x["runs"])
            .sort_values(["win_rate", "wins"], ascending=[False, False])
            .reset_index()
        )
        summaries[entity] = summary
    return summaries


def to_csv_download(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def format_probability(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.1%}"


def format_decimal(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.2f}"


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #f5f6f0;
            color: #17211c;
        }
        .block-container {
            max-width: 1320px;
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background: #17211c;
        }
        [data-testid="stSidebar"] * {
            color: #f5f6f0;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: #17211c;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d9ddcf;
            border-left: 4px solid #2f6b4f;
            border-radius: 8px;
            padding: 0.8rem 1rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #d9ddcf;
            border-radius: 8px;
            overflow: hidden;
        }
        .status-line {
            color: #536057;
            font-size: 0.95rem;
            margin-top: -0.6rem;
            margin-bottom: 1rem;
        }
        .decision-note {
            background: #fffaf0;
            border: 1px solid #d9c38a;
            border-left: 4px solid #b68a35;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            color: #493a18;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_seed_data(database_url: str) -> None:
    counts = table_counts(database_url)
    if counts["historical"] == 0 or counts["current"] == 0:
        sample_history = SETTINGS.sample_historical_csv
        sample_current = SETTINGS.sample_current_csv
        if sample_history.exists() and sample_current.exists():
            seed_database_from_samples(database_url, sample_history, sample_current)


@st.cache_data(ttl=60 * 60)
def load_database(database_url: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    ensure_seed_data(database_url)
    history_df = read_races(database_url, TABLES["historical"])
    current_df = read_races(database_url, TABLES["current"])
    status_df = ingestion_status(database_url)
    counts = table_counts(database_url)
    return history_df, current_df, status_df, counts


def render_sidebar(database_url: str) -> None:
    st.sidebar.title("Racing Intelligence")
    st.sidebar.caption("MySQL-ready model workspace")
    st.sidebar.text_input("Database", value=database_url, disabled=True)

    if st.sidebar.button("Reload database", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.button("Reset sample data", use_container_width=True):
        seed_database_from_samples(database_url)
        st.cache_data.clear()
        st.rerun()

    provider = os.getenv("HORSE_API_PROVIDER", "sample")
    st.sidebar.divider()
    st.sidebar.metric("API provider", provider)
    st.sidebar.caption("Run data_pipeline.py hourly to keep the database fresh.")


def render_metrics(history_df: pd.DataFrame, current_df: pd.DataFrame, counts: Dict[str, int], status_df: pd.DataFrame) -> None:
    winner_rate = "-"
    if "finishing_position" in history_df.columns and not history_df.empty:
        winner_rate = format_probability((pd.to_numeric(history_df["finishing_position"], errors="coerce") == 1).mean())

    last_refresh = "-"
    if not status_df.empty:
        last_refresh = str(status_df.iloc[0]["ingested_at"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historical runs", f"{counts['historical']:,}")
    col2.metric("Current runners", f"{counts['current']:,}")
    col3.metric("Historical win rate", winner_rate)
    col4.metric("Last database refresh", last_refresh)


def render_predictions(scored_df: pd.DataFrame) -> None:
    display_df = scored_df.copy()
    display_df["race_date"] = display_df["race_date"].dt.strftime("%Y-%m-%d")
    display_df["win_probability"] = display_df["win_probability"].map(format_probability)
    display_df["implied_probability"] = display_df["implied_probability"].map(format_probability)
    display_df["value_edge"] = display_df["value_edge"].map(format_probability)
    display_df["model_odds"] = display_df["model_odds"].map(format_decimal)

    columns = [
        "race_date",
        "track",
        "horse",
        "jockey",
        "trainer",
        "odds",
        "model_odds",
        "win_probability",
        "implied_probability",
        "value_edge",
        "suggested_rank",
    ]
    st.dataframe(display_df[columns], use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Racing Intelligence", layout="wide")
    apply_styles()
    render_sidebar(DATABASE_URL)

    history_raw, current_raw, status_df, counts = load_database(DATABASE_URL)
    missing_history = validate_columns(history_raw, REQUIRED_HISTORICAL_COLUMNS)
    missing_current = validate_columns(current_raw, REQUIRED_CURRENT_COLUMNS)

    st.title("Racing Intelligence")
    st.markdown(
        '<div class="status-line">Database-led race analysis, model scoring, and value ranking for upcoming runners.</div>',
        unsafe_allow_html=True,
    )

    render_metrics(history_raw, current_raw, counts, status_df)

    if missing_history or missing_current:
        st.error(f"Database schema is missing columns. Historical: {missing_history}; current: {missing_current}")
        return

    history_features = build_feature_table(history_raw)
    try:
        model_result = train_model(history_features)
    except ValueError as exc:
        st.warning(str(exc))
        st.markdown(
            '<div class="decision-note">Add more historical results to the SQL database, then reload. The model retrains from the latest stored data.</div>',
            unsafe_allow_html=True,
        )
        return

    scored_df = score_current_races(model_result, current_raw)
    entity_tables = summarize_entities(history_features)

    tab_predictions, tab_racecard, tab_model, tab_data = st.tabs(
        ["Predictions", "Race Card", "Model", "Data"]
    )

    with tab_predictions:
        tracks = sorted([track for track in scored_df["track"].dropna().unique()])
        selected_tracks = st.multiselect("Track filter", tracks, default=tracks)
        filtered = scored_df[scored_df["track"].isin(selected_tracks)] if selected_tracks else scored_df
        render_predictions(filtered)
        st.download_button(
            "Download predictions",
            data=to_csv_download(filtered),
            file_name="horse_race_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab_racecard:
        racecard = add_scoring_features(current_raw)
        racecard["race_date"] = racecard["race_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            racecard[
                [
                    "race_date",
                    "track",
                    "distance",
                    "surface",
                    "horse",
                    "jockey",
                    "trainer",
                    "odds",
                    "draw",
                    "weather",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tab_model:
        col1, col2, col3 = st.columns(3)
        col1.metric("Training rows", f"{model_result.training_rows:,}")
        col2.metric("Features used", len(model_result.feature_columns))
        col3.metric("Winner rate", format_probability(model_result.winner_rate))

        trend_tabs = st.tabs(["Jockeys", "Trainers", "Owners"])
        for tab, key in zip(trend_tabs, ["jockey", "trainer", "owner"]):
            with tab:
                table = entity_tables[key].copy()
                table["win_rate"] = table["win_rate"].map(format_probability)
                table["avg_odds"] = table["avg_odds"].map(format_decimal)
                st.dataframe(table.head(25), use_container_width=True, hide_index=True)

    with tab_data:
        st.subheader("Ingestion status")
        st.dataframe(status_df, use_container_width=True, hide_index=True)
        st.markdown(
            '<div class="decision-note">This app reads from the SQL database only. Use the hourly ingestion worker to update the tables from an API, and the model will retrain from those newer rows on reload.</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
