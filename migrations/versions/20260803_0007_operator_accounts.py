"""Add operator accounts.

Revision ID: 20260803_0007
Revises: 20260803_0006
Create Date: 2026-08-03
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision = "20260803_0007"
down_revision = "20260803_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "operator_accounts" in inspector.get_table_names():
        return

    op.create_table(
        "operator_accounts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("account_key", sa.String(length=120), nullable=False, unique=True),
        sa.Column("display_name", sa.String(length=160), nullable=False),
        sa.Column("email", sa.String(length=254), nullable=True),
        sa.Column("roles", sa.Text(), nullable=False),
        sa.Column("token_sha256", sa.String(length=64), nullable=True, unique=True),
        sa.Column("status", sa.String(length=40), nullable=False, server_default="active"),
        sa.Column("privacy_acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_authenticated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_operator_accounts_status", "operator_accounts", ["status"])
    op.create_index("ix_operator_accounts_token_sha256", "operator_accounts", ["token_sha256"])


def downgrade() -> None:
    bind = op.get_bind()
    if "operator_accounts" not in inspect(bind).get_table_names():
        return

    op.drop_index("ix_operator_accounts_token_sha256", table_name="operator_accounts")
    op.drop_index("ix_operator_accounts_status", table_name="operator_accounts")
    op.drop_table("operator_accounts")
