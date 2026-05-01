import io
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

REQUIRED_COLUMNS = [
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


def validate_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()

    work_df["race_date"] = pd.to_datetime(work_df["race_date"], errors="coerce")
    work_df["race_month"] = work_df["race_date"].dt.month
    work_df["race_day_of_week"] = work_df["race_date"].dt.dayofweek

    work_df["is_winner"] = (work_df["finishing_position"] == 1).astype(int)
    work_df["implied_probability"] = 1 / (work_df["odds"].replace(0, np.nan))

    columns = REQUIRED_COLUMNS + [col for col in OPTIONAL_COLUMNS if col in work_df.columns]
    columns += ["race_month", "race_day_of_week", "implied_probability", "is_winner"]

    return work_df[columns]


def train_model(df: pd.DataFrame) -> ModelResult:
    features = [col for col in df.columns if col != "is_winner"]

    numeric_features = [
        col
        for col in features
        if pd.api.types.is_numeric_dtype(df[col]) and col not in ["finishing_position"]
    ]
    categorical_features = [
        col for col in features if col not in numeric_features and col != "race_date"
    ]

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

    X = df[features].drop(columns=["race_date"], errors="ignore")
    y = df["is_winner"]
    model.fit(X, y)

    return ModelResult(model=model, feature_columns=list(X.columns))


def score_current_races(model_result: ModelResult, races_df: pd.DataFrame) -> pd.DataFrame:
    score_df = races_df.copy()
    score_df["race_date"] = pd.to_datetime(score_df["race_date"], errors="coerce")
    score_df["race_month"] = score_df["race_date"].dt.month
    score_df["race_day_of_week"] = score_df["race_date"].dt.dayofweek
    score_df["implied_probability"] = 1 / (score_df["odds"].replace(0, np.nan))

    X = score_df.reindex(columns=model_result.feature_columns)
    score_df["win_probability"] = model_result.model.predict_proba(X)[:, 1]
    score_df["suggested_rank"] = score_df["win_probability"].rank(ascending=False, method="min")

    return score_df.sort_values(["suggested_rank", "win_probability"], ascending=[True, False])


def summarize_entities(history_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    jockey_summary = (
        history_df.groupby("jockey", dropna=False)
        .agg(runs=("horse", "count"), wins=("is_winner", "sum"), avg_odds=("odds", "mean"))
        .assign(win_rate=lambda x: x["wins"] / x["runs"])
        .sort_values("win_rate", ascending=False)
        .reset_index()
    )

    owner_summary = (
        history_df.groupby("owner", dropna=False)
        .agg(runs=("horse", "count"), wins=("is_winner", "sum"), avg_odds=("odds", "mean"))
        .assign(win_rate=lambda x: x["wins"] / x["runs"])
        .sort_values("win_rate", ascending=False)
        .reset_index()
    )

    return {"jockey": jockey_summary, "owner": owner_summary}


def to_csv_download(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def main() -> None:
    st.set_page_config(page_title="Horse Racing Intelligence", layout="wide")
    st.title("🐎 Horse Racing Intelligence App")
    st.caption("Train on historical race and betting data, then score upcoming races.")

    st.markdown(
        """
        Upload two CSV files:
        1. **Historical data** (past races + outcomes + your betting history fields)
        2. **Current races** you want to score
        """
    )

    historical_file = st.file_uploader("Historical data CSV", type=["csv"])
    current_file = st.file_uploader("Current race card CSV", type=["csv"])

    with st.expander("Expected columns"):
        st.write("Required:", REQUIRED_COLUMNS)
        st.write("Optional:", OPTIONAL_COLUMNS)

    if historical_file is None:
        st.info("Upload your historical dataset to get started.")
        return

    history_raw = pd.read_csv(historical_file)
    missing = validate_columns(history_raw)
    if missing:
        st.error(f"Historical file is missing required columns: {missing}")
        return

    history_features = build_feature_table(history_raw)
    st.success(f"Loaded {len(history_features)} historical records.")

    model_result = train_model(history_features)
    entity_tables = summarize_entities(history_features)

    st.subheader("Top Jockey Trends")
    st.dataframe(entity_tables["jockey"].head(15), use_container_width=True)

    st.subheader("Top Owner Trends")
    st.dataframe(entity_tables["owner"].head(15), use_container_width=True)

    if current_file is None:
        st.warning("Upload current race card CSV to generate predictions.")
        return

    current_df = pd.read_csv(current_file)
    missing_current = [col for col in REQUIRED_COLUMNS if col not in current_df.columns]
    if missing_current:
        st.error(f"Current file is missing required columns: {missing_current}")
        return

    scored_df = score_current_races(model_result, current_df)
    st.subheader("Predicted Race Rankings")
    st.dataframe(
        scored_df[
            [
                "horse",
                "jockey",
                "owner",
                "track",
                "odds",
                "win_probability",
                "suggested_rank",
            ]
        ],
        use_container_width=True,
    )

    st.download_button(
        "Download predictions",
        data=to_csv_download(scored_df),
        file_name="horse_race_predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
