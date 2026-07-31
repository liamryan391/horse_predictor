from __future__ import annotations

import io
import os
from typing import Dict, List

import pandas as pd
import streamlit as st

from prediction_model import (
    add_scoring_features,
    build_feature_table,
    evaluate_model,
    score_current_races,
    summarize_entities,
    train_model,
)
from racing_storage import (
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


def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
    return [col for col in required_columns if col not in df.columns]


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
    evaluation = evaluate_model(history_features)

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
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Training rows", f"{model_result.training_rows:,}")
        col2.metric("Features used", len(model_result.feature_columns))
        col3.metric("Winner rate", format_probability(model_result.winner_rate))
        col4.metric("Holdout Brier", format_decimal(evaluation.metrics.get("runner_brier_score")))

        st.subheader("Model evaluation")
        if evaluation.status != "ok":
            st.warning(evaluation.message or "Waiting for enough historical races to evaluate.")
        else:
            eval_cols = st.columns(4)
            eval_cols[0].metric("Validation races", f"{evaluation.validation_races:,}")
            eval_cols[1].metric("Top-pick win rate", format_probability(evaluation.metrics.get("top_pick_win_rate")))
            eval_cols[2].metric("Market top-pick rate", format_probability(evaluation.metrics.get("market_top_pick_win_rate")))
            eval_cols[3].metric("Fixed-stake ROI", format_probability(evaluation.metrics.get("fixed_stake_roi")))

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
