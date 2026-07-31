from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Engine

from settings import BASE_DIR, resolve_database_url

RACE_COLUMNS = [
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

NUMERIC_COLUMNS = [
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

TABLES = {"historical": "races_historical", "current": "races_current"}
VALID_TABLES = set(TABLES.values())


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def get_engine(database_url: str | Path | None = None) -> Engine:
    return create_engine(resolve_database_url(database_url), future=True, pool_pre_ping=True)


def build_metadata() -> MetaData:
    metadata = MetaData()
    for table_name in VALID_TABLES:
        Table(
            table_name,
            metadata,
            Column("race_date", Date, index=True),
            Column("track", String(120), index=True),
            Column("distance", Float),
            Column("surface", String(80)),
            Column("horse", String(160), index=True),
            Column("jockey", String(160)),
            Column("owner", String(160)),
            Column("trainer", String(160)),
            Column("odds", Float),
            Column("finishing_position", Float),
            Column("horse_age", Float),
            Column("horse_weight", Float),
            Column("draw", Float),
            Column("speed_rating", Float),
            Column("class_rating", Float),
            Column("days_since_last_run", Float),
            Column("past_bets_count", Float),
            Column("past_bets_profit", Float),
            Column("weather", String(120)),
            Column("ingested_at", DateTime(timezone=True)),
            Column("source", String(80)),
        )

    Table(
        "ingestion_log",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("table_name", String(80), nullable=False),
        Column("source", String(80), nullable=False),
        Column("row_count", Integer, nullable=False),
        Column("status", String(40), nullable=False),
        Column("message", Text),
        Column("ingested_at", DateTime(timezone=True), nullable=False),
    )
    return metadata


METADATA = build_metadata()


def init_db(database_url: str | Path | None = None) -> None:
    engine = get_engine(database_url)
    METADATA.create_all(engine)


def normalize_race_frame(df: pd.DataFrame, current_mode: bool = False) -> pd.DataFrame:
    work_df = df.copy()
    for column in RACE_COLUMNS:
        if column not in work_df.columns:
            work_df[column] = None

    work_df["race_date"] = pd.to_datetime(work_df["race_date"], errors="coerce").dt.date
    for column in NUMERIC_COLUMNS:
        work_df[column] = pd.to_numeric(work_df[column], errors="coerce")

    if current_mode:
        work_df["finishing_position"] = work_df["finishing_position"].fillna(0)

    return work_df[RACE_COLUMNS]


def write_races(
    database_url: str | Path | None,
    table_name: str,
    df: pd.DataFrame,
    source: str,
    replace: bool = True,
) -> int:
    if table_name not in VALID_TABLES:
        raise ValueError(f"Unsupported table: {table_name}")

    init_db(database_url)
    engine = get_engine(database_url)
    table = METADATA.tables[table_name]
    log_table = METADATA.tables["ingestion_log"]
    current_mode = table_name == TABLES["current"]

    write_df = normalize_race_frame(df, current_mode=current_mode)
    write_df["ingested_at"] = utc_now()
    write_df["source"] = source

    with engine.begin() as conn:
        if replace:
            conn.execute(delete(table))
        if not write_df.empty:
            conn.execute(insert(table), write_df.to_dict(orient="records"))
        conn.execute(
            insert(log_table),
            {
                "table_name": table_name,
                "source": source,
                "row_count": len(write_df),
                "status": "success",
                "message": None,
                "ingested_at": utc_now(),
            },
        )

    return len(write_df)


def read_races(database_url: str | Path | None, table_name: str) -> pd.DataFrame:
    if table_name not in VALID_TABLES:
        raise ValueError(f"Unsupported table: {table_name}")

    init_db(database_url)
    engine = get_engine(database_url)
    with engine.connect() as conn:
        return pd.read_sql_query(text(f"SELECT * FROM {table_name}"), conn)


def table_counts(database_url: str | Path | None) -> Dict[str, int]:
    init_db(database_url)
    engine = get_engine(database_url)
    with engine.connect() as conn:
        return {
            name: int(conn.execute(select(func.count()).select_from(METADATA.tables[table])).scalar_one())
            for name, table in TABLES.items()
        }


def ingestion_status(database_url: str | Path | None) -> pd.DataFrame:
    init_db(database_url)
    engine = get_engine(database_url)
    with engine.connect() as conn:
        return pd.read_sql_query(
            text(
                """
                SELECT table_name, source, row_count, status, message, ingested_at
                FROM ingestion_log
                ORDER BY id DESC
                LIMIT 20
                """
            ),
            conn,
        )


def seed_database_from_samples(
    database_url: str | Path | None = None,
    historical_csv: str | Path = "sample_historical_data.csv",
    current_csv: str | Path = "sample_current_races.csv",
) -> Dict[str, int]:
    historical_df = pd.read_csv(historical_csv)
    current_df = pd.read_csv(current_csv)
    return {
        "historical": write_races(database_url, TABLES["historical"], historical_df, source="sample"),
        "current": write_races(database_url, TABLES["current"], current_df, source="sample"),
    }
