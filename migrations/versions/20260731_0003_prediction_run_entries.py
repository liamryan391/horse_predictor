"""Add prediction run entries.

Revision ID: 20260731_0003
Revises: 20260731_0002
Create Date: 2026-07-31
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision = "20260731_0003"
down_revision = "20260731_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if "prediction_run_entries" in inspect(bind).get_table_names():
        return
    op.create_table(
        "prediction_run_entries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("prediction_run_id", sa.Integer(), sa.ForeignKey("prediction_runs.id"), nullable=False),
        sa.Column("race_date", sa.Date(), nullable=True),
        sa.Column("track", sa.String(length=120), nullable=True),
        sa.Column("distance", sa.Float(), nullable=True),
        sa.Column("surface", sa.String(length=80), nullable=True),
        sa.Column("horse", sa.String(length=160), nullable=True),
        sa.Column("jockey", sa.String(length=160), nullable=True),
        sa.Column("owner", sa.String(length=160), nullable=True),
        sa.Column("trainer", sa.String(length=160), nullable=True),
        sa.Column("market_odds", sa.Float(), nullable=True),
        sa.Column("win_probability", sa.Float(), nullable=True),
        sa.Column("model_odds", sa.Float(), nullable=True),
        sa.Column("value_edge", sa.Float(), nullable=True),
        sa.Column("suggested_rank", sa.Float(), nullable=True),
        sa.Column("field_size", sa.Float(), nullable=True),
        sa.Column("odds_rank", sa.Float(), nullable=True),
        sa.Column("raw_features", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_prediction_run_entries_run_rank",
        "prediction_run_entries",
        ["prediction_run_id", "suggested_rank"],
    )
    op.create_index(
        "ix_prediction_run_entries_date_track",
        "prediction_run_entries",
        ["race_date", "track"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    if "prediction_run_entries" not in inspect(bind).get_table_names():
        return
    op.drop_index("ix_prediction_run_entries_date_track", table_name="prediction_run_entries")
    op.drop_index("ix_prediction_run_entries_run_rank", table_name="prediction_run_entries")
    op.drop_table("prediction_run_entries")
