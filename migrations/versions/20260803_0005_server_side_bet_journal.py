"""Add server-side bet journal fields.

Revision ID: 20260803_0005
Revises: 20260731_0004
Create Date: 2026-08-03
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision = "20260803_0005"
down_revision = "20260731_0004"
branch_labels = None
depends_on = None


JOURNAL_COLUMNS = {
    "account_key": sa.Column("account_key", sa.String(length=120), nullable=False, server_default="local"),
    "prediction_run_id": sa.Column("prediction_run_id", sa.Integer(), sa.ForeignKey("prediction_runs.id"), nullable=True),
    "prediction_run_entry_id": sa.Column(
        "prediction_run_entry_id",
        sa.Integer(),
        sa.ForeignKey("prediction_run_entries.id"),
        nullable=True,
    ),
    "model_version_id": sa.Column("model_version_id", sa.Integer(), sa.ForeignKey("model_versions.id"), nullable=True),
    "horse": sa.Column("horse", sa.String(length=160), nullable=True),
    "track": sa.Column("track", sa.String(length=120), nullable=True),
    "race_date": sa.Column("race_date", sa.Date(), nullable=True),
    "closing_odds_decimal": sa.Column("closing_odds_decimal", sa.Float(), nullable=True),
    "updated_at": sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
}

JOURNAL_INDEXES = {
    "ix_user_bets_account_status": ("account_key", "status"),
    "ix_user_bets_race_date_track": ("race_date", "track"),
    "ix_user_bets_prediction_run": ("prediction_run_id",),
}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "user_bets" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("user_bets")}
    with op.batch_alter_table("user_bets") as batch:
        if "race_entry_id" in existing_columns:
            batch.alter_column("race_entry_id", existing_type=sa.Integer(), nullable=True)
        for name, column in JOURNAL_COLUMNS.items():
            if name not in existing_columns:
                batch.add_column(column)

    existing_indexes = {index["name"] for index in inspect(bind).get_indexes("user_bets")}
    for name, columns in JOURNAL_INDEXES.items():
        if name not in existing_indexes and all(column in existing_columns | set(JOURNAL_COLUMNS) for column in columns):
            op.create_index(name, "user_bets", list(columns))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "user_bets" not in inspector.get_table_names():
        return

    existing_indexes = {index["name"] for index in inspector.get_indexes("user_bets")}
    for name in reversed(tuple(JOURNAL_INDEXES)):
        if name in existing_indexes:
            op.drop_index(name, table_name="user_bets")

    existing_columns = {column["name"] for column in inspect(bind).get_columns("user_bets")}
    if "race_entry_id" in existing_columns:
        bind.execute(sa.text("DELETE FROM user_bets WHERE race_entry_id IS NULL"))

    with op.batch_alter_table("user_bets") as batch:
        for name in reversed(tuple(JOURNAL_COLUMNS)):
            if name in existing_columns:
                batch.drop_column(name)
        if "race_entry_id" in existing_columns:
            batch.alter_column("race_entry_id", existing_type=sa.Integer(), nullable=False)
