from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from math import ceil, isfinite
from pathlib import Path
import pickle
import re
from typing import Any, Dict, List
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from racing_storage import RACE_COLUMNS

RACE_GROUP_COLUMNS = ["race_date", "track", "distance"]
RESULT_LEAKAGE_COLUMNS = {"finishing_position", "is_winner"}
MODEL_EXCLUDED_COLUMNS = RESULT_LEAKAGE_COLUMNS | {"race_date"}
EVALUATION_EPSILON = 1e-6
MODEL_ARTIFACT_FORMAT = "horse-predictor-model-artifact-v1"


@dataclass
class ModelResult:
    model: Pipeline
    feature_columns: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    training_rows: int
    winner_rate: float
    training_start: str | None
    training_end: str | None


@dataclass
class ModelArtifactRef:
    uri: str
    sha256: str
    feature_schema_hash: str
    code_commit_sha: str | None
    metadata: Dict[str, Any]


@dataclass
class EvaluationResult:
    status: str
    message: str | None
    training_rows: int
    validation_rows: int
    training_races: int
    validation_races: int
    evaluation_start: str | None
    evaluation_end: str | None
    metrics: Dict[str, float | int | None]
    leakage_features: List[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "trainingRows": self.training_rows,
            "validationRows": self.validation_rows,
            "trainingRaces": self.training_races,
            "validationRaces": self.validation_races,
            "evaluationStart": self.evaluation_start,
            "evaluationEnd": self.evaluation_end,
            "metrics": self.metrics,
            "leakageFeatures": self.leakage_features,
        }


def clean_metric(value) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if isfinite(value) else None
    return value


def feature_schema_payload(
    feature_columns: list[str],
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> dict[str, list[str]]:
    return {
        "feature_columns": list(feature_columns),
        "numeric_features": list(numeric_features or []),
        "categorical_features": list(categorical_features or []),
    }


def feature_schema_hash(
    feature_columns: list[str],
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> str:
    payload = feature_schema_payload(feature_columns, numeric_features, categorical_features)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_feature_schema_hash(model_result: ModelResult) -> str:
    return feature_schema_hash(
        model_result.feature_columns,
        model_result.numeric_features,
        model_result.categorical_features,
    )


def model_artifact_metadata(model_result: ModelResult, code_commit_sha: str | None = None) -> dict[str, Any]:
    schema_hash = model_feature_schema_hash(model_result)
    return {
        "format": MODEL_ARTIFACT_FORMAT,
        "algorithm": "LogisticRegression",
        "feature_columns": model_result.feature_columns,
        "numeric_features": model_result.numeric_features,
        "categorical_features": model_result.categorical_features,
        "feature_schema_hash": schema_hash,
        "training_rows": model_result.training_rows,
        "winner_rate": model_result.winner_rate,
        "training_start": model_result.training_start,
        "training_end": model_result.training_end,
        "code_commit_sha": code_commit_sha,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def _safe_artifact_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip(".-")
    return stem[:120] or "model-artifact"


def _unique_artifact_path(artifact_dir: Path, stem: str, suffix: str = ".pkl") -> Path:
    path = artifact_dir / f"{stem}{suffix}"
    if not path.exists():
        return path

    for counter in range(1, 1000):
        candidate = artifact_dir / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not choose a unique model artifact filename for {stem}.")


def artifact_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(url2pathname(parsed.path)).resolve()
    if parsed.scheme:
        raise ValueError(f"Unsupported model artifact URI scheme: {parsed.scheme}")
    return Path(uri).resolve()


def save_model_artifact(
    model_result: ModelResult,
    artifact_dir: str | Path,
    name: str | None = None,
    code_commit_sha: str | None = None,
) -> ModelArtifactRef:
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    metadata = model_artifact_metadata(model_result, code_commit_sha=code_commit_sha)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    base_name = name or "logistic-regression"
    stem = _safe_artifact_stem(f"{base_name}-{timestamp}-{metadata['feature_schema_hash'][:12]}")
    path = _unique_artifact_path(artifact_path, stem)

    payload = {
        "format": MODEL_ARTIFACT_FORMAT,
        "metadata": metadata,
        "model_result": model_result,
    }
    blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    artifact_sha256 = hashlib.sha256(blob).hexdigest()
    path.write_bytes(blob)

    sidecar_metadata = {**metadata, "artifact_sha256": artifact_sha256, "artifact_uri": path.resolve().as_uri()}
    path.with_suffix(".json").write_text(json.dumps(sidecar_metadata, indent=2, sort_keys=True), encoding="utf-8")

    return ModelArtifactRef(
        uri=path.resolve().as_uri(),
        sha256=artifact_sha256,
        feature_schema_hash=metadata["feature_schema_hash"],
        code_commit_sha=code_commit_sha,
        metadata=sidecar_metadata,
    )


def load_model_artifact(
    artifact_uri: str,
    expected_sha256: str | None = None,
    expected_feature_schema_hash: str | None = None,
) -> ModelResult:
    path = artifact_uri_to_path(artifact_uri)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact was not found: {artifact_uri}")

    blob = path.read_bytes()
    artifact_sha256 = hashlib.sha256(blob).hexdigest()
    if expected_sha256 and artifact_sha256 != expected_sha256:
        raise ValueError("Model artifact checksum does not match registry metadata.")

    payload = pickle.loads(blob)
    if not isinstance(payload, dict) or payload.get("format") != MODEL_ARTIFACT_FORMAT:
        raise ValueError("Model artifact format is not supported.")

    model_result = payload.get("model_result")
    if not isinstance(model_result, ModelResult):
        raise ValueError("Model artifact does not contain a valid ModelResult.")

    actual_schema_hash = model_feature_schema_hash(model_result)
    artifact_schema_hash = (payload.get("metadata") or {}).get("feature_schema_hash")
    if artifact_schema_hash and artifact_schema_hash != actual_schema_hash:
        raise ValueError("Model artifact feature schema metadata does not match its model payload.")
    if expected_feature_schema_hash and expected_feature_schema_hash != actual_schema_hash:
        raise ValueError("Model artifact feature schema does not match registry metadata.")

    assert_no_leakage(model_result.feature_columns)
    return model_result


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


def iso_date(value) -> str | None:
    if pd.isna(value):
        return None
    if hasattr(value, "date"):
        value = value.date()
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def date_range(df: pd.DataFrame) -> tuple[str | None, str | None]:
    dates = pd.to_datetime(df["race_date"], errors="coerce").dropna()
    if dates.empty:
        return None, None
    return iso_date(dates.min()), iso_date(dates.max())


def add_pre_race_features(work_df: pd.DataFrame) -> pd.DataFrame:
    work_df = work_df.copy()
    work_df["race_month"] = work_df["race_date"].dt.month
    work_df["race_day_of_week"] = work_df["race_date"].dt.dayofweek
    work_df["implied_probability"] = 1 / work_df["odds"].replace(0, np.nan)

    race_group = work_df.groupby(RACE_GROUP_COLUMNS, dropna=False)
    work_df["field_size"] = race_group["horse"].transform("count")
    work_df["odds_rank"] = race_group["odds"].rank(ascending=True, method="min")
    work_df["relative_speed_rating"] = work_df["speed_rating"] - race_group["speed_rating"].transform("mean")
    work_df["relative_class_rating"] = work_df["class_rating"] - race_group["class_rating"].transform("mean")
    return work_df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    work_df = coerce_race_frame(df)
    work_df = work_df.dropna(subset=["finishing_position"]).copy()

    work_df = add_pre_race_features(work_df)
    work_df["is_winner"] = (work_df["finishing_position"] == 1).astype(int)

    columns = RACE_COLUMNS + [
        "race_month",
        "race_day_of_week",
        "implied_probability",
        "field_size",
        "odds_rank",
        "relative_speed_rating",
        "relative_class_rating",
        "is_winner",
    ]
    return work_df[columns]


def leakage_features(feature_columns: list[str]) -> list[str]:
    return sorted(set(feature_columns) & RESULT_LEAKAGE_COLUMNS)


def assert_no_leakage(feature_columns: list[str]) -> None:
    leaked = leakage_features(feature_columns)
    if leaked:
        raise ValueError(f"Model feature set includes result-only leakage columns: {', '.join(leaked)}")


def train_model(df: pd.DataFrame) -> ModelResult:
    if df.empty:
        raise ValueError("No historical rows are available for training.")
    if df["is_winner"].nunique() < 2:
        raise ValueError("Historical data needs at least one winner and one non-winner.")

    features = [col for col in df.columns if col not in MODEL_EXCLUDED_COLUMNS]
    assert_no_leakage(features)
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
    training_start, training_end = date_range(df)

    return ModelResult(
        model=model,
        feature_columns=list(X.columns),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        training_rows=len(df),
        winner_rate=float(y.mean()),
        training_start=training_start,
        training_end=training_end,
    )


def predict_probabilities(model_result: ModelResult, feature_df: pd.DataFrame) -> np.ndarray:
    X = feature_df.reindex(columns=model_result.feature_columns)
    return model_result.model.predict_proba(X)[:, 1]


def add_scoring_features(races_df: pd.DataFrame) -> pd.DataFrame:
    score_df = coerce_race_frame(races_df)
    return add_pre_race_features(score_df)


def score_current_races(model_result: ModelResult, races_df: pd.DataFrame) -> pd.DataFrame:
    score_df = add_scoring_features(races_df)
    score_df["win_probability"] = predict_probabilities(model_result, score_df)
    score_df["model_odds"] = np.where(score_df["win_probability"] > 0, 1 / score_df["win_probability"], np.nan)
    score_df["value_edge"] = score_df["win_probability"] - score_df["implied_probability"]

    race_group = ["race_date", "track", "distance"]
    score_df["suggested_rank"] = score_df.groupby(race_group, dropna=False)["win_probability"].rank(
        ascending=False,
        method="min",
    )
    return score_df.sort_values(race_group + ["suggested_rank", "win_probability"], ascending=[True, True, True, True, False])


def race_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=RACE_GROUP_COLUMNS)
    return (
        df[RACE_GROUP_COLUMNS]
        .drop_duplicates()
        .sort_values(RACE_GROUP_COLUMNS, kind="mergesort")
        .reset_index(drop=True)
    )


def race_key_mask(df: pd.DataFrame, keys: pd.DataFrame) -> pd.Series:
    if keys.empty:
        return pd.Series(False, index=df.index)
    selected = keys.copy()
    selected["_selected"] = True
    merged = df[RACE_GROUP_COLUMNS].merge(selected, on=RACE_GROUP_COLUMNS, how="left")
    return pd.Series(merged["_selected"].eq(True).to_numpy(), index=df.index)


def clipped_probabilities(values: pd.Series, fallback: float) -> np.ndarray:
    probabilities = pd.to_numeric(values, errors="coerce").fillna(fallback).to_numpy(dtype=float)
    return np.clip(probabilities, EVALUATION_EPSILON, 1 - EVALUATION_EPSILON)


def calibration_error(actual: pd.Series, probabilities: np.ndarray, bins: int = 5) -> float | None:
    if len(actual) == 0:
        return None
    calibration_df = pd.DataFrame({"actual": actual.to_numpy(dtype=int), "probability": probabilities})
    bin_count = min(bins, len(calibration_df))
    try:
        calibration_df["bin"] = pd.qcut(calibration_df["probability"], q=bin_count, duplicates="drop")
    except ValueError:
        calibration_df["bin"] = "all"
    grouped = calibration_df.groupby("bin", observed=True).agg(
        actual_rate=("actual", "mean"),
        expected_rate=("probability", "mean"),
        rows=("actual", "count"),
    )
    if grouped.empty:
        return None
    gaps = (grouped["actual_rate"] - grouped["expected_rate"]).abs()
    return float(np.average(gaps, weights=grouped["rows"]))


def top_pick_metrics(scored_df: pd.DataFrame, score_column: str) -> tuple[float | None, float | None]:
    if scored_df.empty:
        return None, None

    top_pick_wins = []
    winner_ranks = []
    for _, race_df in scored_df.groupby(RACE_GROUP_COLUMNS, dropna=False):
        scores = pd.to_numeric(race_df[score_column], errors="coerce").fillna(-np.inf)
        top_pick = race_df.loc[scores.idxmax()]
        top_pick_wins.append(int(top_pick["is_winner"] == 1))

        ranks = scores.rank(ascending=False, method="min")
        for row_index in race_df.index[race_df["is_winner"] == 1]:
            winner_ranks.append(float(ranks.loc[row_index]))

    top_pick_win_rate = float(np.mean(top_pick_wins)) if top_pick_wins else None
    mean_winner_rank = float(np.mean(winner_ranks)) if winner_ranks else None
    return top_pick_win_rate, mean_winner_rank


def fixed_stake_backtest(scored_df: pd.DataFrame) -> dict[str, float | int | None]:
    if scored_df.empty:
        return {"fixed_stake_bets": 0, "fixed_stake_profit": None, "fixed_stake_roi": None}

    picks = []
    for _, race_df in scored_df.groupby(RACE_GROUP_COLUMNS, dropna=False):
        scores = pd.to_numeric(race_df["win_probability"], errors="coerce").fillna(-np.inf)
        picks.append(race_df.loc[scores.idxmax()])
    top_picks = pd.DataFrame(picks)
    value_picks = top_picks[pd.to_numeric(top_picks["value_edge"], errors="coerce") > 0].copy()

    if value_picks.empty:
        return {"fixed_stake_bets": 0, "fixed_stake_profit": None, "fixed_stake_roi": None}

    decimal_odds = pd.to_numeric(value_picks["odds"], errors="coerce")
    profit = np.where(value_picks["is_winner"] == 1, decimal_odds - 1, -1)
    profit = pd.Series(profit).replace([np.inf, -np.inf], np.nan).dropna()
    if profit.empty:
        return {"fixed_stake_bets": 0, "fixed_stake_profit": None, "fixed_stake_roi": None}

    total_profit = float(profit.sum())
    bets = int(len(profit))
    return {
        "fixed_stake_bets": bets,
        "fixed_stake_profit": total_profit,
        "fixed_stake_roi": total_profit / bets if bets else None,
    }


def skipped_evaluation(message: str, df: pd.DataFrame | None = None) -> EvaluationResult:
    training_start = training_end = None
    if df is not None and not df.empty and "race_date" in df.columns:
        training_start, training_end = date_range(df)
    return EvaluationResult(
        status="skipped",
        message=message,
        training_rows=0 if df is None else len(df),
        validation_rows=0,
        training_races=0,
        validation_races=0,
        evaluation_start=training_start,
        evaluation_end=training_end,
        metrics={},
        leakage_features=[],
    )


def evaluate_model(df: pd.DataFrame, validation_fraction: float = 0.25) -> EvaluationResult:
    if df.empty:
        return skipped_evaluation("No historical rows are available for evaluation.", df)
    if df["is_winner"].nunique() < 2:
        return skipped_evaluation("Historical data needs at least one winner and one non-winner for evaluation.", df)

    keys = race_keys(df)
    if len(keys) < 2:
        return skipped_evaluation("At least two chronological races are needed for holdout evaluation.", df)

    validation_race_count = max(1, ceil(len(keys) * validation_fraction))
    validation_race_count = min(validation_race_count, len(keys) - 1)
    train_keys = keys.iloc[:-validation_race_count]
    validation_keys = keys.iloc[-validation_race_count:]
    train_df = df[race_key_mask(df, train_keys)].copy()
    validation_df = df[race_key_mask(df, validation_keys)].copy()

    try:
        model_result = train_model(train_df)
    except ValueError as exc:
        return skipped_evaluation(str(exc), df)

    predicted = clipped_probabilities(pd.Series(predict_probabilities(model_result, validation_df)), model_result.winner_rate)
    market = clipped_probabilities(validation_df["implied_probability"], model_result.winner_rate)
    actual = validation_df["is_winner"].astype(int)

    scored_df = validation_df.copy()
    scored_df["win_probability"] = predicted
    scored_df["market_probability"] = market
    scored_df["model_odds"] = 1 / scored_df["win_probability"]
    scored_df["value_edge"] = scored_df["win_probability"] - scored_df["implied_probability"]

    model_top_pick_win_rate, mean_winner_rank = top_pick_metrics(scored_df, "win_probability")
    market_top_pick_win_rate, market_mean_winner_rank = top_pick_metrics(scored_df, "market_probability")
    metrics: Dict[str, float | int | None] = {
        "runner_log_loss": clean_metric(log_loss(actual, predicted, labels=[0, 1])),
        "runner_brier_score": clean_metric(brier_score_loss(actual, predicted)),
        "market_log_loss": clean_metric(log_loss(actual, market, labels=[0, 1])),
        "market_brier_score": clean_metric(brier_score_loss(actual, market)),
        "calibration_mae": clean_metric(calibration_error(actual, predicted)),
        "top_pick_win_rate": clean_metric(model_top_pick_win_rate),
        "market_top_pick_win_rate": clean_metric(market_top_pick_win_rate),
        "mean_winner_rank": clean_metric(mean_winner_rank),
        "market_mean_winner_rank": clean_metric(market_mean_winner_rank),
    }
    metrics.update({key: clean_metric(value) for key, value in fixed_stake_backtest(scored_df).items()})
    evaluation_start, evaluation_end = date_range(validation_df)

    return EvaluationResult(
        status="ok",
        message=None,
        training_rows=len(train_df),
        validation_rows=len(validation_df),
        training_races=len(train_keys),
        validation_races=len(validation_keys),
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        metrics=metrics,
        leakage_features=leakage_features(model_result.feature_columns),
    )


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
