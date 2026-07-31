"""Add ingestion job locks.

Revision ID: 20260731_0002
Revises: 20260731_0001
Create Date: 2026-07-31
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision = "20260731_0002"
down_revision = "20260731_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if "job_locks" in inspect(bind).get_table_names():
        return
    op.create_table(
        "job_locks",
        sa.Column("lock_name", sa.String(length=120), primary_key=True),
        sa.Column("owner", sa.String(length=160), nullable=False),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
    )
    op.create_index("ix_job_locks_expires_at", "job_locks", ["expires_at"])


def downgrade() -> None:
    bind = op.get_bind()
    if "job_locks" not in inspect(bind).get_table_names():
        return
    op.drop_index("ix_job_locks_expires_at", table_name="job_locks")
    op.drop_table("job_locks")
