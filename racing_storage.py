from __future__ import annotations

import json
import hashlib
from datetime import date, datetime, timedelta, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Dict

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

from race_enrichment import MODEL_ENRICHMENT_COLUMNS, distance_to_yards
from schema import METADATA, NUMERIC_COLUMNS, RACE_COLUMNS, RACE_IDENTITY_COLUMNS, TABLES, VALID_TABLES
from settings import resolve_database_url

BET_STATUSES = {"open", "won", "lost", "void"}
DEFAULT_BET_ACCOUNT_KEY = "local"


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


def _as_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def _iso_value(value) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _int_or_none(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_safe_value(value) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (float, int)):
        return value if isfinite(float(value)) else None
    return value


def _payload_value(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return default


def _payload_has(payload: dict[str, Any], *keys: str) -> bool:
    return any(key in payload for key in keys)


def _trim_text(value: Any, max_length: int | None = None) -> str | None:
    if value is None:
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    return text_value[:max_length] if max_length else text_value


def _required_text(value: Any, field_name: str, max_length: int | None = None) -> str:
    text_value = _trim_text(value, max_length=max_length)
    if text_value is None:
        raise ValueError(f"{field_name} is required.")
    return text_value


def _positive_number(value: Any, field_name: str, minimum: float = 0) -> float:
    numeric = _float_or_none(value)
    if numeric is None or numeric <= minimum:
        raise ValueError(f"{field_name} must be greater than {minimum:g}.")
    return numeric


def _bet_status(value: Any, default: str = "open") -> str:
    status = _trim_text(value, max_length=80) or default
    status = status.lower()
    if status not in BET_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(sorted(BET_STATUSES))}.")
    return status


def _bet_profit_loss(status: str, stake: float | None, odds_decimal: float | None) -> float | None:
    if status == "open":
        return None
    if status == "void":
        return 0.0
    if stake is None:
        return None
    if status == "lost":
        return -stake
    if odds_decimal is None:
        return None
    return stake * (odds_decimal - 1)


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
        if column == "distance":
            work_df[column] = work_df[column].map(distance_to_yards)
        else:
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


def _model_metric_sample_size(metric_name: str, evaluation_result: Any) -> int | None:
    if metric_name == "fixed_stake_bets":
        return evaluation_result.validation_races
    if metric_name.startswith("fixed_stake"):
        bets = evaluation_result.metrics.get("fixed_stake_bets")
        return int(bets) if bets is not None else None
    if "top_pick" in metric_name or "winner_rank" in metric_name:
        return evaluation_result.validation_races
    return evaluation_result.validation_rows


def _model_payload(row: dict[str, Any], metrics: dict[str, float | int | None]) -> dict[str, Any]:
    feature_set = row.get("feature_set")
    try:
        features = json.loads(feature_set) if feature_set else []
    except json.JSONDecodeError:
        features = []

    return {
        "id": row["id"],
        "name": row["name"],
        "algorithm": row["algorithm"],
        "status": row["status"],
        "featureCount": len(features),
        "trainingStart": _iso_value(row.get("training_start")),
        "trainingEnd": _iso_value(row.get("training_end")),
        "artifactUri": row.get("artifact_uri"),
        "artifactSha256": row.get("artifact_sha256"),
        "featureSchemaHash": row.get("feature_schema_hash"),
        "codeCommitSha": row.get("code_commit_sha"),
        "artifactReady": bool(row.get("artifact_uri") and row.get("artifact_sha256") and row.get("feature_schema_hash")),
        "createdAt": _iso_value(row.get("created_at")),
        "updatedAt": _iso_value(row.get("updated_at")),
        "metrics": metrics,
    }


def _feature_schema_hash(
    feature_columns: list[str],
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> str:
    encoded = json.dumps(
        {
            "categorical_features": list(categorical_features or []),
            "feature_columns": list(feature_columns),
            "numeric_features": list(numeric_features or []),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metrics_for_model_ids(conn, model_ids: list[int]) -> dict[int, dict[str, float | int | None]]:
    if not model_ids:
        return {}

    eval_table = METADATA.tables["model_evaluation_results"]
    rows = conn.execute(
        select(eval_table.c.model_version_id, eval_table.c.metric_name, eval_table.c.metric_value).where(
            eval_table.c.model_version_id.in_(model_ids)
        )
    ).all()
    metrics: dict[int, dict[str, float | int | None]] = {model_id: {} for model_id in model_ids}
    for model_id, metric_name, metric_value in rows:
        metrics[int(model_id)][metric_name] = metric_value
    return metrics


def record_model_evaluation_snapshot(
    database_url: str | Path | None,
    model_result: Any,
    evaluation_result: Any,
    status: str = "candidate",
    name: str | None = None,
    artifact_uri: str | None = None,
    artifact_sha256: str | None = None,
    feature_schema_hash: str | None = None,
    code_commit_sha: str | None = None,
) -> dict[str, Any]:
    init_db(database_url)
    engine = get_engine(database_url)
    model_table = METADATA.tables["model_versions"]
    eval_table = METADATA.tables["model_evaluation_results"]
    now = utc_now()
    feature_columns = list(getattr(model_result, "feature_columns", []))
    model_name = name or f"logistic-regression-{now.strftime('%Y%m%d%H%M%S')}"
    schema_hash = feature_schema_hash or _feature_schema_hash(
        feature_columns,
        list(getattr(model_result, "numeric_features", [])),
        list(getattr(model_result, "categorical_features", [])),
    )

    try:
        with engine.begin() as conn:
            result = conn.execute(
                insert(model_table),
                {
                    "name": model_name,
                    "algorithm": "LogisticRegression",
                    "feature_set": json.dumps(feature_columns, separators=(",", ":")),
                    "training_start": _as_date(getattr(model_result, "training_start", None)),
                    "training_end": _as_date(getattr(model_result, "training_end", None)),
                    "artifact_uri": artifact_uri,
                    "artifact_sha256": artifact_sha256,
                    "feature_schema_hash": schema_hash,
                    "code_commit_sha": code_commit_sha,
                    "status": status,
                    "created_at": now,
                    "updated_at": now,
                },
            )
            model_version_id = int(result.inserted_primary_key[0])

            metric_records = []
            for metric_name, metric_value in evaluation_result.metrics.items():
                if metric_value is None:
                    continue
                metric_records.append(
                    {
                        "model_version_id": model_version_id,
                        "metric_name": metric_name,
                        "metric_value": float(metric_value),
                        "sample_size": _model_metric_sample_size(metric_name, evaluation_result),
                        "evaluation_start": _as_date(evaluation_result.evaluation_start),
                        "evaluation_end": _as_date(evaluation_result.evaluation_end),
                        "created_at": now,
                    }
                )
            if metric_records:
                conn.execute(insert(eval_table), metric_records)

            row = conn.execute(select(model_table).where(model_table.c.id == model_version_id)).mappings().one()
            metrics = _metrics_for_model_ids(conn, [model_version_id]).get(model_version_id, {})
            return _model_payload(dict(row), metrics)
    finally:
        engine.dispose()


def read_model_version(database_url: str | Path | None, model_version_id: int) -> dict[str, Any] | None:
    init_db(database_url)
    engine = get_engine(database_url)
    model_table = METADATA.tables["model_versions"]
    try:
        with engine.connect() as conn:
            row = conn.execute(select(model_table).where(model_table.c.id == model_version_id)).mappings().first()
            if not row:
                return None
            metrics = _metrics_for_model_ids(conn, [model_version_id]).get(model_version_id, {})
            return _model_payload(dict(row), metrics)
    finally:
        engine.dispose()


def read_model_registry(database_url: str | Path | None, limit: int = 20, offset: int = 0) -> dict[str, Any]:
    init_db(database_url)
    engine = get_engine(database_url)
    model_table = METADATA.tables["model_versions"]
    try:
        with engine.connect() as conn:
            total = int(conn.execute(select(func.count()).select_from(model_table)).scalar_one())
            rows = (
                conn.execute(
                    select(model_table)
                    .order_by(model_table.c.id.desc())
                    .limit(limit)
                    .offset(offset)
                )
                .mappings()
                .all()
            )
            model_ids = [int(row["id"]) for row in rows]
            metrics = _metrics_for_model_ids(conn, model_ids)
            models = [_model_payload(dict(row), metrics.get(int(row["id"]), {})) for row in rows]
            return {
                "models": models,
                "page": {"limit": limit, "offset": offset, "returned": len(models), "total": total},
            }
    finally:
        engine.dispose()


def approve_model_version(database_url: str | Path | None, model_version_id: int) -> dict[str, Any] | None:
    init_db(database_url)
    engine = get_engine(database_url)
    model_table = METADATA.tables["model_versions"]
    now = utc_now()
    try:
        with engine.begin() as conn:
            row = conn.execute(select(model_table).where(model_table.c.id == model_version_id)).mappings().first()
            if not row:
                return None
            conn.execute(
                update(model_table)
                .where(model_table.c.status == "approved", model_table.c.id != model_version_id)
                .values(status="superseded", updated_at=now)
            )
            conn.execute(update(model_table).where(model_table.c.id == model_version_id).values(status="approved", updated_at=now))
            updated = conn.execute(select(model_table).where(model_table.c.id == model_version_id)).mappings().one()
            metrics = _metrics_for_model_ids(conn, [model_version_id]).get(model_version_id, {})
            return _model_payload(dict(updated), metrics)
    finally:
        engine.dispose()


def read_latest_approved_model_version(database_url: str | Path | None) -> dict[str, Any] | None:
    init_db(database_url)
    engine = get_engine(database_url)
    model_table = METADATA.tables["model_versions"]
    try:
        with engine.connect() as conn:
            row = (
                conn.execute(
                    select(model_table)
                    .where(model_table.c.status == "approved")
                    .order_by(model_table.c.id.desc())
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if not row:
                return None
            metrics = _metrics_for_model_ids(conn, [int(row["id"])]).get(int(row["id"]), {})
            return _model_payload(dict(row), metrics)
    finally:
        engine.dispose()


def _prediction_feature_payload(row: dict[str, Any]) -> dict[str, Any]:
    feature_columns = [
        *MODEL_ENRICHMENT_COLUMNS,
        "race_month",
        "race_day_of_week",
        "implied_probability",
        "field_size",
        "odds_rank",
        "relative_speed_rating",
        "relative_class_rating",
        "draw",
        "speed_rating",
        "class_rating",
        "days_since_last_run",
        "horse_age",
        "horse_weight",
        "weather",
    ]
    return {column: _json_safe_value(row.get(column)) for column in feature_columns if column in row}


def _prediction_entry_record(prediction_run_id: int, row: dict[str, Any], now: datetime) -> dict[str, Any]:
    return {
        "prediction_run_id": prediction_run_id,
        "race_date": _as_date(_json_safe_value(row.get("race_date"))),
        "track": _json_safe_value(row.get("track")),
        "distance": _float_or_none(row.get("distance")),
        "surface": _json_safe_value(row.get("surface")),
        "horse": _json_safe_value(row.get("horse")),
        "jockey": _json_safe_value(row.get("jockey")),
        "owner": _json_safe_value(row.get("owner")),
        "trainer": _json_safe_value(row.get("trainer")),
        "market_odds": _float_or_none(row.get("odds")),
        "win_probability": _float_or_none(row.get("win_probability")),
        "model_odds": _float_or_none(row.get("model_odds")),
        "value_edge": _float_or_none(row.get("value_edge")),
        "suggested_rank": _float_or_none(row.get("suggested_rank")),
        "field_size": _float_or_none(row.get("field_size")),
        "odds_rank": _float_or_none(row.get("odds_rank")),
        "raw_features": json.dumps(_prediction_feature_payload(row), separators=(",", ":"), allow_nan=False),
        "created_at": now,
    }


def _prediction_entry_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "race_date": _iso_value(row.get("race_date")),
        "track": row.get("track"),
        "distance": row.get("distance"),
        "surface": row.get("surface"),
        "horse": row.get("horse"),
        "jockey": row.get("jockey"),
        "owner": row.get("owner"),
        "trainer": row.get("trainer"),
        "odds": row.get("market_odds"),
        "win_probability": row.get("win_probability"),
        "model_odds": row.get("model_odds"),
        "value_edge": row.get("value_edge"),
        "suggested_rank": row.get("suggested_rank"),
        "field_size": row.get("field_size"),
        "odds_rank": row.get("odds_rank"),
    }


def _prediction_run_payload(conn, row: dict[str, Any]) -> dict[str, Any]:
    entry_table = METADATA.tables["prediction_run_entries"]
    run_id = int(row["id"])
    runner_count = int(
        conn.execute(
            select(func.count()).select_from(entry_table).where(entry_table.c.prediction_run_id == run_id)
        ).scalar_one()
    )
    top_entry = (
        conn.execute(
            select(entry_table)
            .where(entry_table.c.prediction_run_id == run_id)
            .order_by(entry_table.c.suggested_rank.asc(), entry_table.c.win_probability.desc(), entry_table.c.id.asc())
            .limit(1)
        )
        .mappings()
        .first()
    )
    return {
        "id": run_id,
        "modelVersionId": row.get("model_version_id"),
        "raceId": row.get("race_id"),
        "runAt": _iso_value(row.get("run_at")),
        "source": row.get("source"),
        "notes": row.get("notes"),
        "runnerCount": runner_count,
        "topRunner": top_entry.get("horse") if top_entry else None,
        "topWinProbability": top_entry.get("win_probability") if top_entry else None,
        "topValueEdge": top_entry.get("value_edge") if top_entry else None,
    }


def record_prediction_run(
    database_url: str | Path | None,
    predictions_df: pd.DataFrame,
    source: str = "api",
    model_version_id: int | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if predictions_df.empty:
        raise ValueError("No predictions are available to record.")

    init_db(database_url)
    engine = get_engine(database_url)
    run_table = METADATA.tables["prediction_runs"]
    entry_table = METADATA.tables["prediction_run_entries"]
    now = utc_now()
    row_records = [record for record in predictions_df.to_dict(orient="records")]

    try:
        with engine.begin() as conn:
            result = conn.execute(
                insert(run_table),
                {
                    "model_version_id": model_version_id,
                    "race_id": None,
                    "run_at": now,
                    "source": source,
                    "notes": notes,
                },
            )
            prediction_run_id = int(result.inserted_primary_key[0])
            entries = [_prediction_entry_record(prediction_run_id, record, now) for record in row_records]
            conn.execute(insert(entry_table), entries)

            run_row = conn.execute(select(run_table).where(run_table.c.id == prediction_run_id)).mappings().one()
            entry_rows = (
                conn.execute(
                    select(entry_table)
                    .where(entry_table.c.prediction_run_id == prediction_run_id)
                    .order_by(entry_table.c.suggested_rank.asc(), entry_table.c.win_probability.desc(), entry_table.c.id.asc())
                )
                .mappings()
                .all()
            )
            return {
                "run": _prediction_run_payload(conn, dict(run_row)),
                "entries": [_prediction_entry_payload(dict(row)) for row in entry_rows],
                "page": {
                    "limit": len(entry_rows),
                    "offset": 0,
                    "returned": len(entry_rows),
                    "total": len(entry_rows),
                },
            }
    finally:
        engine.dispose()


def read_prediction_runs(database_url: str | Path | None, limit: int = 20, offset: int = 0) -> dict[str, Any]:
    init_db(database_url)
    engine = get_engine(database_url)
    run_table = METADATA.tables["prediction_runs"]
    try:
        with engine.connect() as conn:
            total = int(conn.execute(select(func.count()).select_from(run_table)).scalar_one())
            rows = (
                conn.execute(
                    select(run_table)
                    .order_by(run_table.c.id.desc())
                    .limit(limit)
                    .offset(offset)
                )
                .mappings()
                .all()
            )
            runs = [_prediction_run_payload(conn, dict(row)) for row in rows]
            return {
                "runs": runs,
                "page": {"limit": limit, "offset": offset, "returned": len(runs), "total": total},
            }
    finally:
        engine.dispose()


def read_prediction_run(
    database_url: str | Path | None,
    prediction_run_id: int,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any] | None:
    init_db(database_url)
    engine = get_engine(database_url)
    run_table = METADATA.tables["prediction_runs"]
    entry_table = METADATA.tables["prediction_run_entries"]
    try:
        with engine.connect() as conn:
            run_row = conn.execute(select(run_table).where(run_table.c.id == prediction_run_id)).mappings().first()
            if not run_row:
                return None
            total = int(
                conn.execute(
                    select(func.count()).select_from(entry_table).where(entry_table.c.prediction_run_id == prediction_run_id)
                ).scalar_one()
            )
            rows = (
                conn.execute(
                    select(entry_table)
                    .where(entry_table.c.prediction_run_id == prediction_run_id)
                    .order_by(entry_table.c.suggested_rank.asc(), entry_table.c.win_probability.desc(), entry_table.c.id.asc())
                    .limit(limit)
                    .offset(offset)
                )
                .mappings()
                .all()
            )
            entries = [_prediction_entry_payload(dict(row)) for row in rows]
            return {
                "run": _prediction_run_payload(conn, dict(run_row)),
                "entries": entries,
                "page": {"limit": limit, "offset": offset, "returned": len(entries), "total": total},
            }
    finally:
        engine.dispose()


def _bet_payload(row: dict[str, Any]) -> dict[str, Any]:
    race_entry_id = row.get("race_entry_id")
    horse = row.get("horse") or (f"Race entry #{race_entry_id}" if race_entry_id else "Unmatched runner")
    return {
        "id": int(row["id"]),
        "accountKey": row.get("account_key") or DEFAULT_BET_ACCOUNT_KEY,
        "raceEntryId": race_entry_id,
        "predictionRunId": row.get("prediction_run_id"),
        "predictionRunEntryId": row.get("prediction_run_entry_id"),
        "modelVersionId": row.get("model_version_id"),
        "horse": horse,
        "track": row.get("track"),
        "raceDate": _iso_value(row.get("race_date")),
        "betType": row.get("bet_type"),
        "stake": row.get("stake"),
        "odds": row.get("odds_decimal"),
        "closingOdds": row.get("closing_odds_decimal"),
        "status": row.get("status"),
        "createdAt": _iso_value(row.get("placed_at")),
        "updatedAt": _iso_value(row.get("updated_at") or row.get("placed_at")),
        "settledAt": _iso_value(row.get("settled_at")),
        "profitLoss": row.get("profit_loss"),
        "notes": row.get("notes"),
    }


def _bet_insert_record(payload: dict[str, Any], account_key: str, now: datetime) -> dict[str, Any]:
    status = _bet_status(_payload_value(payload, "status"), default="open")
    stake = _positive_number(_payload_value(payload, "stake"), "stake")
    odds_decimal = _positive_number(_payload_value(payload, "odds", "oddsDecimal"), "odds", minimum=1)
    settled_at = _as_utc_datetime(_payload_value(payload, "settledAt", "settled_at"))
    if status != "open" and settled_at is None:
        settled_at = now

    return {
        "account_key": _required_text(account_key, "account_key", max_length=120),
        "race_entry_id": _int_or_none(_payload_value(payload, "raceEntryId", "race_entry_id")),
        "prediction_run_id": _int_or_none(_payload_value(payload, "predictionRunId", "prediction_run_id")),
        "prediction_run_entry_id": _int_or_none(_payload_value(payload, "predictionRunEntryId", "prediction_run_entry_id")),
        "model_version_id": _int_or_none(_payload_value(payload, "modelVersionId", "model_version_id")),
        "horse": _required_text(_payload_value(payload, "horse"), "horse", max_length=160),
        "track": _trim_text(_payload_value(payload, "track"), max_length=120),
        "race_date": _as_date(_payload_value(payload, "raceDate", "race_date")),
        "bet_type": _trim_text(_payload_value(payload, "betType", "bet_type"), max_length=80) or "win",
        "stake": stake,
        "odds_decimal": odds_decimal,
        "closing_odds_decimal": _positive_number(
            _payload_value(payload, "closingOdds", "closing_odds_decimal"),
            "closingOdds",
            minimum=1,
        )
        if _payload_value(payload, "closingOdds", "closing_odds_decimal") is not None
        else None,
        "status": status,
        "placed_at": _as_utc_datetime(_payload_value(payload, "createdAt", "placedAt", "placed_at")) or now,
        "settled_at": settled_at,
        "profit_loss": _bet_profit_loss(status, stake, odds_decimal),
        "notes": _trim_text(_payload_value(payload, "notes"), max_length=2000),
        "updated_at": now,
    }


def record_bet_journal_entry(
    database_url: str | Path | None,
    payload: dict[str, Any],
    account_key: str = DEFAULT_BET_ACCOUNT_KEY,
) -> dict[str, Any]:
    init_db(database_url)
    engine = get_engine(database_url)
    bet_table = METADATA.tables["user_bets"]
    now = utc_now()
    record = _bet_insert_record(payload, account_key, now)

    try:
        with engine.begin() as conn:
            result = conn.execute(insert(bet_table), record)
            bet_id = int(result.inserted_primary_key[0])
            row = conn.execute(select(bet_table).where(bet_table.c.id == bet_id)).mappings().one()
            return _bet_payload(dict(row))
    finally:
        engine.dispose()


def read_bet_journal(
    database_url: str | Path | None,
    account_key: str = DEFAULT_BET_ACCOUNT_KEY,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    init_db(database_url)
    engine = get_engine(database_url)
    bet_table = METADATA.tables["user_bets"]
    filters = [bet_table.c.account_key == account_key]
    if status:
        filters.append(bet_table.c.status == _bet_status(status))

    try:
        with engine.connect() as conn:
            total = int(conn.execute(select(func.count()).select_from(bet_table).where(*filters)).scalar_one())
            rows = (
                conn.execute(
                    select(bet_table)
                    .where(*filters)
                    .order_by(bet_table.c.placed_at.desc(), bet_table.c.id.desc())
                    .limit(limit)
                    .offset(offset)
                )
                .mappings()
                .all()
            )
            bets = [_bet_payload(dict(row)) for row in rows]
            return {"bets": bets, "page": {"limit": limit, "offset": offset, "returned": len(bets), "total": total}}
    finally:
        engine.dispose()


def update_bet_journal_entry(
    database_url: str | Path | None,
    bet_id: int,
    payload: dict[str, Any],
    account_key: str = DEFAULT_BET_ACCOUNT_KEY,
) -> dict[str, Any] | None:
    init_db(database_url)
    engine = get_engine(database_url)
    bet_table = METADATA.tables["user_bets"]
    now = utc_now()

    try:
        with engine.begin() as conn:
            existing = (
                conn.execute(
                    select(bet_table).where(bet_table.c.id == bet_id, bet_table.c.account_key == account_key)
                )
                .mappings()
                .first()
            )
            if not existing:
                return None

            existing_record = dict(existing)
            updates: dict[str, Any] = {}
            if _payload_has(payload, "horse"):
                updates["horse"] = _required_text(payload["horse"], "horse", max_length=160)
            if _payload_has(payload, "track"):
                updates["track"] = _trim_text(payload.get("track"), max_length=120)
            if _payload_has(payload, "raceDate", "race_date"):
                updates["race_date"] = _as_date(_payload_value(payload, "raceDate", "race_date"))
            if _payload_has(payload, "betType", "bet_type"):
                updates["bet_type"] = _trim_text(_payload_value(payload, "betType", "bet_type"), max_length=80) or "win"
            if _payload_has(payload, "stake"):
                updates["stake"] = _positive_number(payload["stake"], "stake")
            if _payload_has(payload, "odds", "oddsDecimal"):
                updates["odds_decimal"] = _positive_number(_payload_value(payload, "odds", "oddsDecimal"), "odds", minimum=1)
            if _payload_has(payload, "closingOdds", "closing_odds_decimal"):
                value = _payload_value(payload, "closingOdds", "closing_odds_decimal")
                updates["closing_odds_decimal"] = None if value is None else _positive_number(value, "closingOdds", minimum=1)
            if _payload_has(payload, "status"):
                updates["status"] = _bet_status(payload["status"])
            if _payload_has(payload, "settledAt", "settled_at"):
                updates["settled_at"] = _as_utc_datetime(_payload_value(payload, "settledAt", "settled_at"))
            if _payload_has(payload, "notes"):
                updates["notes"] = _trim_text(payload.get("notes"), max_length=2000)
            if _payload_has(payload, "raceEntryId", "race_entry_id"):
                updates["race_entry_id"] = _int_or_none(_payload_value(payload, "raceEntryId", "race_entry_id"))
            if _payload_has(payload, "predictionRunId", "prediction_run_id"):
                updates["prediction_run_id"] = _int_or_none(_payload_value(payload, "predictionRunId", "prediction_run_id"))
            if _payload_has(payload, "predictionRunEntryId", "prediction_run_entry_id"):
                updates["prediction_run_entry_id"] = _int_or_none(_payload_value(payload, "predictionRunEntryId", "prediction_run_entry_id"))
            if _payload_has(payload, "modelVersionId", "model_version_id"):
                updates["model_version_id"] = _int_or_none(_payload_value(payload, "modelVersionId", "model_version_id"))

            status = updates.get("status", existing_record["status"])
            stake = updates.get("stake", existing_record["stake"])
            odds_decimal = updates.get("odds_decimal", existing_record["odds_decimal"])
            updates["profit_loss"] = _bet_profit_loss(status, stake, odds_decimal)
            if status == "open":
                updates["settled_at"] = None
            elif "settled_at" not in updates and existing_record.get("settled_at") is None:
                updates["settled_at"] = now
            updates["updated_at"] = now

            conn.execute(update(bet_table).where(bet_table.c.id == bet_id, bet_table.c.account_key == account_key).values(updates))
            row = conn.execute(select(bet_table).where(bet_table.c.id == bet_id)).mappings().one()
            return _bet_payload(dict(row))
    finally:
        engine.dispose()


def delete_bet_journal_entry(
    database_url: str | Path | None,
    bet_id: int,
    account_key: str = DEFAULT_BET_ACCOUNT_KEY,
) -> bool:
    init_db(database_url)
    engine = get_engine(database_url)
    bet_table = METADATA.tables["user_bets"]
    try:
        with engine.begin() as conn:
            result = conn.execute(delete(bet_table).where(bet_table.c.id == bet_id, bet_table.c.account_key == account_key))
            return bool(result.rowcount)
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


def provider_freshness_report(database_url: str | Path | None) -> list[dict[str, Any]]:
    init_db(database_url)
    engine = get_engine(database_url)
    try:
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    select(METADATA.tables["api_ingestion_runs"])
                    .order_by(
                        METADATA.tables["api_ingestion_runs"].c.provider.asc(),
                        METADATA.tables["api_ingestion_runs"].c.target_table.asc(),
                        METADATA.tables["api_ingestion_runs"].c.completed_at.desc(),
                        METADATA.tables["api_ingestion_runs"].c.id.desc(),
                    )
                )
                .mappings()
                .all()
            )
    finally:
        engine.dispose()

    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["provider"], row["target_table"])
        if key in latest:
            continue
        latest[key] = {
            "provider": row["provider"],
            "tableName": row["target_table"],
            "status": row["status"],
            "rowCount": row["row_count"],
            "completedAt": _iso_value(row.get("completed_at")),
            "message": row.get("message"),
        }
    return list(latest.values())


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
