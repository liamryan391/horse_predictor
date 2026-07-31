from __future__ import annotations

import math
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from prediction_model import build_feature_table, score_current_races, summarize_entities, train_model
from racing_storage import (
    TABLES,
    ingestion_status,
    read_races,
    seed_database_from_samples,
    table_counts,
)
from settings import get_settings

SETTINGS = get_settings()
SETTINGS.validate_runtime()
DATABASE_URL = SETTINGS.database_url

app = FastAPI(title="Horse Predictor API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.backend_cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_seed_data() -> None:
    counts = table_counts(DATABASE_URL)
    if counts["historical"] == 0 or counts["current"] == 0:
        sample_history = SETTINGS.sample_historical_csv
        sample_current = SETTINGS.sample_current_csv
        if sample_history.exists() and sample_current.exists():
            seed_database_from_samples(DATABASE_URL, sample_history, sample_current)


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None
    return value


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: clean_value(value) for key, value in row.items()} for row in df.to_dict(orient="records")]


def database_summary() -> dict[str, Any]:
    return SETTINGS.database_summary()


def load_model_bundle() -> tuple[pd.DataFrame, pd.DataFrame, Any, pd.DataFrame]:
    ensure_seed_data()
    history_df = read_races(DATABASE_URL, TABLES["historical"])
    current_df = read_races(DATABASE_URL, TABLES["current"])
    history_features = build_feature_table(history_df)
    try:
        model = train_model(history_features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return history_df, current_df, model, history_features


@app.get("/api/health")
def health() -> dict[str, Any]:
    counts = table_counts(DATABASE_URL)
    return {"status": "ok", "database": database_summary(), "counts": counts}


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    ensure_seed_data()
    counts = table_counts(DATABASE_URL)
    status_df = ingestion_status(DATABASE_URL)
    last_refresh = None if status_df.empty else clean_value(status_df.iloc[0]["ingested_at"])
    return {
        "database": database_summary(),
        "historicalRuns": counts["historical"],
        "currentRunners": counts["current"],
        "lastRefresh": last_refresh,
    }


@app.get("/api/predictions")
def predictions() -> dict[str, Any]:
    _, current_df, model, _ = load_model_bundle()
    scored_df = score_current_races(model, current_df)
    return {"predictions": records(scored_df)}


@app.get("/api/race-card")
def race_card() -> dict[str, Any]:
    ensure_seed_data()
    current_df = read_races(DATABASE_URL, TABLES["current"])
    return {"raceCard": records(current_df)}


@app.get("/api/model")
def model_status() -> dict[str, Any]:
    _, _, model, history_features = load_model_bundle()
    return {
        "trainingRows": model.training_rows,
        "winnerRate": model.winner_rate,
        "featureCount": len(model.feature_columns),
        "features": model.feature_columns,
        "historicalRows": len(history_features),
    }


@app.get("/api/trends")
def trends() -> dict[str, Any]:
    _, _, _, history_features = load_model_bundle()
    entity_tables = summarize_entities(history_features)
    return {name: records(table.head(25)) for name, table in entity_tables.items()}


@app.get("/api/ingestion-status")
def ingestion() -> dict[str, Any]:
    ensure_seed_data()
    return {"ingestion": records(ingestion_status(DATABASE_URL))}
