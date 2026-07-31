from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from racing_storage import RACE_COLUMNS


@dataclass
class ModelResult:
    model: Pipeline
    feature_columns: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    training_rows: int
    winner_rate: float


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
