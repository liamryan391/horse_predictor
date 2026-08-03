"""Add admin audit events.

Revision ID: 20260803_0006
Revises: 20260803_0005
Create Date: 2026-08-03
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision = "20260803_0006"
down_revision = "20260803_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "admin_audit_events" in inspector.get_table_names():
        return

    op.create_table(
        "admin_audit_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("actor", sa.String(length=160), nullable=False),
        sa.Column("roles", sa.Text(), nullable=True),
        sa.Column("action", sa.String(length=160), nullable=False),
        sa.Column("resource_type", sa.String(length=120), nullable=False),
        sa.Column("resource_id", sa.String(length=120), nullable=True),
        sa.Column("request_id", sa.String(length=120), nullable=True),
        sa.Column("status", sa.String(length=40), nullable=False, server_default="success"),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_admin_audit_events_created", "admin_audit_events", ["created_at"])
    op.create_index("ix_admin_audit_events_actor", "admin_audit_events", ["actor"])
    op.create_index("ix_admin_audit_events_action", "admin_audit_events", ["action"])


def downgrade() -> None:
    bind = op.get_bind()
    if "admin_audit_events" not in inspect(bind).get_table_names():
        return

    op.drop_index("ix_admin_audit_events_action", table_name="admin_audit_events")
    op.drop_index("ix_admin_audit_events_actor", table_name="admin_audit_events")
    op.drop_index("ix_admin_audit_events_created", table_name="admin_audit_events")
    op.drop_table("admin_audit_events")
