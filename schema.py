from __future__ import annotations

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)

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
RACE_IDENTITY_COLUMNS = ["race_date", "track", "distance", "horse", "source"]


def timestamp_columns() -> list[Column]:
    return [
        Column("created_at", DateTime(timezone=True)),
        Column("updated_at", DateTime(timezone=True)),
    ]


def provider_columns(entity_name: str) -> list[Column]:
    return [
        Column("provider", String(80), nullable=False, default="manual"),
        Column(f"provider_{entity_name}_id", String(160)),
    ]


def add_compatibility_race_tables(metadata: MetaData) -> None:
    for table_name in sorted(VALID_TABLES):
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
            Column("source", String(80), index=True),
            UniqueConstraint(
                "race_date",
                "track",
                "distance",
                "horse",
                "source",
                name=f"uq_{table_name}_runner_source",
            ),
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


def add_normalized_tables(metadata: MetaData) -> None:
    Table(
        "courses",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        *provider_columns("course"),
        Column("name", String(160), nullable=False),
        Column("country", String(80)),
        *timestamp_columns(),
        UniqueConstraint("provider", "name", name="uq_courses_provider_name"),
        Index("ix_courses_name", "name"),
    )

    for table_name, entity_name in [
        ("horses", "horse"),
        ("jockeys", "jockey"),
        ("trainers", "trainer"),
        ("owners", "owner"),
    ]:
        Table(
            table_name,
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            *provider_columns(entity_name),
            Column("name", String(160), nullable=False),
            *timestamp_columns(),
            UniqueConstraint("provider", "name", name=f"uq_{table_name}_provider_name"),
            Index(f"ix_{table_name}_name", "name"),
        )

    Table(
        "race_meetings",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        *provider_columns("meeting"),
        Column("race_date", Date, nullable=False),
        Column("course_id", Integer, ForeignKey("courses.id"), nullable=False),
        *timestamp_columns(),
        UniqueConstraint("provider", "race_date", "course_id", name="uq_race_meetings_provider_date_course"),
        Index("ix_race_meetings_date_course", "race_date", "course_id"),
    )

    Table(
        "races",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        *provider_columns("race"),
        Column("meeting_id", Integer, ForeignKey("race_meetings.id"), nullable=False),
        Column("course_id", Integer, ForeignKey("courses.id"), nullable=False),
        Column("race_date", Date, nullable=False),
        Column("off_time", String(40)),
        Column("distance_yards", Float),
        Column("surface", String(80)),
        Column("going", String(120)),
        Column("race_class", Float),
        Column("raw_payload", Text),
        *timestamp_columns(),
        UniqueConstraint(
            "provider",
            "race_date",
            "course_id",
            "off_time",
            "distance_yards",
            name="uq_races_provider_date_course_time_distance",
        ),
        Index("ix_races_date_course", "race_date", "course_id"),
    )

    Table(
        "race_entries",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        *provider_columns("entry"),
        Column("race_id", Integer, ForeignKey("races.id"), nullable=False),
        Column("horse_id", Integer, ForeignKey("horses.id"), nullable=False),
        Column("jockey_id", Integer, ForeignKey("jockeys.id")),
        Column("trainer_id", Integer, ForeignKey("trainers.id")),
        Column("owner_id", Integer, ForeignKey("owners.id")),
        Column("draw", Float),
        Column("horse_age", Float),
        Column("horse_weight", Float),
        Column("speed_rating", Float),
        Column("class_rating", Float),
        Column("days_since_last_run", Float),
        Column("raw_payload", Text),
        *timestamp_columns(),
        UniqueConstraint("provider", "race_id", "horse_id", name="uq_race_entries_provider_race_horse"),
        Index("ix_race_entries_race_id", "race_id"),
        Index("ix_race_entries_horse_id", "horse_id"),
    )

    Table(
        "historical_results",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("race_entry_id", Integer, ForeignKey("race_entries.id"), nullable=False, unique=True),
        Column("finishing_position", Float),
        Column("result_status", String(80)),
        Column("source", String(80), nullable=False),
        Column("raw_payload", Text),
        *timestamp_columns(),
    )

    Table(
        "odds_snapshots",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("race_entry_id", Integer, ForeignKey("race_entries.id"), nullable=False),
        Column("provider", String(80), nullable=False),
        Column("odds_decimal", Float),
        Column("captured_at", DateTime(timezone=True), nullable=False),
        Column("source", String(80)),
        Column("raw_payload", Text),
        Index("ix_odds_snapshots_entry_time", "race_entry_id", "captured_at"),
    )

    Table(
        "user_bets",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("account_key", String(120), nullable=False, default="local"),
        Column("race_entry_id", Integer, ForeignKey("race_entries.id")),
        Column("prediction_run_id", Integer, ForeignKey("prediction_runs.id")),
        Column("prediction_run_entry_id", Integer, ForeignKey("prediction_run_entries.id")),
        Column("model_version_id", Integer, ForeignKey("model_versions.id")),
        Column("horse", String(160)),
        Column("track", String(120)),
        Column("race_date", Date),
        Column("bet_type", String(80), nullable=False),
        Column("stake", Float, nullable=False),
        Column("odds_decimal", Float),
        Column("closing_odds_decimal", Float),
        Column("status", String(80), nullable=False, default="open"),
        Column("placed_at", DateTime(timezone=True), nullable=False),
        Column("settled_at", DateTime(timezone=True)),
        Column("profit_loss", Float),
        Column("notes", Text),
        Column("updated_at", DateTime(timezone=True)),
        Index("ix_user_bets_account_status", "account_key", "status"),
        Index("ix_user_bets_race_date_track", "race_date", "track"),
        Index("ix_user_bets_prediction_run", "prediction_run_id"),
    )

    Table(
        "api_ingestion_runs",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("provider", String(80), nullable=False),
        Column("target_table", String(80), nullable=False),
        Column("status", String(40), nullable=False),
        Column("row_count", Integer, nullable=False, default=0),
        Column("started_at", DateTime(timezone=True), nullable=False),
        Column("completed_at", DateTime(timezone=True)),
        Column("message", Text),
        Index("ix_api_ingestion_runs_provider_started", "provider", "started_at"),
    )

    Table(
        "admin_audit_events",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("actor", String(160), nullable=False),
        Column("roles", Text),
        Column("action", String(160), nullable=False),
        Column("resource_type", String(120), nullable=False),
        Column("resource_id", String(120)),
        Column("request_id", String(120)),
        Column("status", String(40), nullable=False, default="success"),
        Column("detail", Text),
        Column("payload_json", Text),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Index("ix_admin_audit_events_created", "created_at"),
        Index("ix_admin_audit_events_actor", "actor"),
        Index("ix_admin_audit_events_action", "action"),
    )

    Table(
        "job_locks",
        metadata,
        Column("lock_name", String(120), primary_key=True),
        Column("owner", String(160), nullable=False),
        Column("acquired_at", DateTime(timezone=True), nullable=False),
        Column("expires_at", DateTime(timezone=True), nullable=False),
        Column("message", Text),
        Index("ix_job_locks_expires_at", "expires_at"),
    )

    Table(
        "model_versions",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("name", String(160), nullable=False),
        Column("algorithm", String(160), nullable=False),
        Column("feature_set", Text),
        Column("training_start", Date),
        Column("training_end", Date),
        Column("artifact_uri", String(512)),
        Column("artifact_sha256", String(64)),
        Column("feature_schema_hash", String(64)),
        Column("code_commit_sha", String(80)),
        Column("status", String(80), nullable=False, default="candidate"),
        *timestamp_columns(),
    )

    Table(
        "prediction_runs",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("model_version_id", Integer, ForeignKey("model_versions.id")),
        Column("race_id", Integer, ForeignKey("races.id")),
        Column("run_at", DateTime(timezone=True), nullable=False),
        Column("source", String(80), nullable=False),
        Column("notes", Text),
    )

    Table(
        "prediction_run_entries",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("prediction_run_id", Integer, ForeignKey("prediction_runs.id"), nullable=False),
        Column("race_date", Date),
        Column("track", String(120)),
        Column("distance", Float),
        Column("surface", String(80)),
        Column("horse", String(160)),
        Column("jockey", String(160)),
        Column("owner", String(160)),
        Column("trainer", String(160)),
        Column("market_odds", Float),
        Column("win_probability", Float),
        Column("model_odds", Float),
        Column("value_edge", Float),
        Column("suggested_rank", Float),
        Column("field_size", Float),
        Column("odds_rank", Float),
        Column("raw_features", Text),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Index("ix_prediction_run_entries_run_rank", "prediction_run_id", "suggested_rank"),
        Index("ix_prediction_run_entries_date_track", "race_date", "track"),
    )

    Table(
        "model_evaluation_results",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("model_version_id", Integer, ForeignKey("model_versions.id"), nullable=False),
        Column("metric_name", String(120), nullable=False),
        Column("metric_value", Float, nullable=False),
        Column("sample_size", Integer),
        Column("evaluation_start", Date),
        Column("evaluation_end", Date),
        Column("created_at", DateTime(timezone=True), nullable=False),
        UniqueConstraint("model_version_id", "metric_name", name="uq_model_eval_model_metric"),
    )


def build_metadata() -> MetaData:
    metadata = MetaData()
    add_compatibility_race_tables(metadata)
    add_normalized_tables(metadata)
    return metadata


METADATA = build_metadata()
