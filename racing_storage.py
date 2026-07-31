from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
from sqlalchemy.exc import IntegrityError

from schema import METADATA, NUMERIC_COLUMNS, RACE_COLUMNS, RACE_IDENTITY_COLUMNS, TABLES, VALID_TABLES
from settings import resolve_database_url


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _as_utc_datetime(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = value
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def get_engine(database_url: str | Path | None = None) -> Engine:
    return create_engine(resolve_database_url(database_url), future=True, pool_pre_ping=True)


def init_db(database_url: str | Path | None = None) -> None:
    engine = get_engine(database_url)
    try:
        METADATA.create_all(engine)
    finally:
        engine.dispose()


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
    message: str | None = None,
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

    try:
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
                    "message": message,
                    "ingested_at": utc_now(),
                },
            )
            run_message = "replace" if replace else "upsert"
            if message:
                run_message = f"{run_message}; {message}"
            insert_ingestion_run(conn, source, table_name, "success", len(records), run_message, started_at)
    finally:
        engine.dispose()

    return len(write_df)


def insert_ingestion_run(conn, provider: str, target_table: str, status: str, row_count: int, message: str | None, started_at: datetime) -> None:
    conn.execute(
        insert(METADATA.tables["api_ingestion_runs"]),
        {
            "provider": provider,
            "target_table": target_table,
            "status": status,
            "row_count": row_count,
            "started_at": started_at,
            "completed_at": utc_now(),
            "message": message,
        },
    )


def record_ingestion_run(
    database_url: str | Path | None,
    provider: str,
    target_table: str,
    status: str,
    row_count: int = 0,
    message: str | None = None,
) -> None:
    init_db(database_url)
    engine = get_engine(database_url)
    try:
        with engine.begin() as conn:
            insert_ingestion_run(conn, provider, target_table, status, row_count, message, utc_now())
    finally:
        engine.dispose()


def try_acquire_job_lock(
    database_url: str | Path | None,
    lock_name: str,
    owner: str,
    ttl_seconds: int,
    message: str | None = None,
) -> bool:
    init_db(database_url)
    engine = get_engine(database_url)
    lock_table = METADATA.tables["job_locks"]
    now = utc_now()
    expires_at = now + timedelta(seconds=ttl_seconds)
    try:
        with engine.begin() as conn:
            row = conn.execute(select(lock_table).where(lock_table.c.lock_name == lock_name)).mappings().first()
            if row:
                current_expiry = _as_utc_datetime(row["expires_at"])
                if current_expiry and current_expiry > now:
                    return False
                result = conn.execute(
                    update(lock_table)
                    .where(lock_table.c.lock_name == lock_name, lock_table.c.expires_at == row["expires_at"])
                    .values(owner=owner, acquired_at=now, expires_at=expires_at, message=message)
                )
                return bool(result.rowcount)
            conn.execute(
                insert(lock_table),
                {
                    "lock_name": lock_name,
                    "owner": owner,
                    "acquired_at": now,
                    "expires_at": expires_at,
                    "message": message,
                },
            )
            return True
    except IntegrityError:
        return False
    finally:
        engine.dispose()


def release_job_lock(database_url: str | Path | None, lock_name: str, owner: str) -> bool:
    init_db(database_url)
    engine = get_engine(database_url)
    lock_table = METADATA.tables["job_locks"]
    try:
        with engine.begin() as conn:
            result = conn.execute(delete(lock_table).where(lock_table.c.lock_name == lock_name, lock_table.c.owner == owner))
            return bool(result.rowcount)
    finally:
        engine.dispose()


def read_races(database_url: str | Path | None, table_name: str) -> pd.DataFrame:
    if table_name not in VALID_TABLES:
        raise ValueError(f"Unsupported table: {table_name}")

    init_db(database_url)
    engine = get_engine(database_url)
    try:
        with engine.connect() as conn:
            return pd.read_sql_query(text(f"SELECT * FROM {table_name}"), conn)
    finally:
        engine.dispose()


def table_counts(database_url: str | Path | None) -> Dict[str, int]:
    init_db(database_url)
    engine = get_engine(database_url)
    try:
        with engine.connect() as conn:
            return {
                name: int(conn.execute(select(func.count()).select_from(METADATA.tables[table])).scalar_one())
                for name, table in TABLES.items()
            }
    finally:
        engine.dispose()


def ingestion_status(database_url: str | Path | None) -> pd.DataFrame:
    init_db(database_url)
    engine = get_engine(database_url)
    try:
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
    finally:
        engine.dispose()


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
