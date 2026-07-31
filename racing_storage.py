from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
from sqlalchemy import (
    and_,
    create_engine,
    delete,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine

from schema import METADATA, NUMERIC_COLUMNS, RACE_COLUMNS, RACE_IDENTITY_COLUMNS, TABLES, VALID_TABLES
from settings import resolve_database_url


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def get_engine(database_url: str | Path | None = None) -> Engine:
    return create_engine(resolve_database_url(database_url), future=True, pool_pre_ping=True)


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


def clean_record(record: dict) -> dict:
    clean = {}
    for key, value in record.items():
        clean[key] = None if pd.isna(value) else value
    return clean


def identity_filter(table, record: dict):
    filters = []
    for column_name in RACE_IDENTITY_COLUMNS:
        column = table.c[column_name]
        value = record.get(column_name)
        filters.append(column.is_(None) if value is None else column == value)
    return and_(*filters)


def upsert_race_records(conn, table, records: list[dict]) -> None:
    for record in records:
        match = identity_filter(table, record)
        exists = conn.execute(select(func.count()).select_from(table).where(match)).scalar_one()
        if exists:
            conn.execute(update(table).where(match).values(record))
        else:
            conn.execute(insert(table), record)


def write_races(
    database_url: str | Path | None,
    table_name: str,
    df: pd.DataFrame,
    source: str,
    replace: bool = False,
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
    records = [clean_record(record) for record in write_df.to_dict(orient="records")]
    started_at = utc_now()

    with engine.begin() as conn:
        if replace:
            conn.execute(delete(table))
            if records:
                conn.execute(insert(table), records)
        elif records:
            upsert_race_records(conn, table, records)
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
        conn.execute(
            insert(METADATA.tables["api_ingestion_runs"]),
            {
                "provider": source,
                "target_table": table_name,
                "status": "success",
                "row_count": len(records),
                "started_at": started_at,
                "completed_at": utc_now(),
                "message": "replace" if replace else "upsert",
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
                SELECT target_table AS table_name, provider AS source, row_count, status, message, completed_at AS ingested_at
                FROM api_ingestion_runs
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
        "historical": write_races(database_url, TABLES["historical"], historical_df, source="sample", replace=True),
        "current": write_races(database_url, TABLES["current"], current_df, source="sample", replace=True),
    }
