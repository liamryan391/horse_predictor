"""Add model artifact integrity metadata.

Revision ID: 20260731_0004
Revises: 20260731_0003
Create Date: 2026-07-31
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision = "20260731_0004"
down_revision = "20260731_0003"
branch_labels = None
depends_on = None


ARTIFACT_COLUMNS = {
    "artifact_sha256": sa.Column("artifact_sha256", sa.String(length=64), nullable=True),
    "feature_schema_hash": sa.Column("feature_schema_hash", sa.String(length=64), nullable=True),
    "code_commit_sha": sa.Column("code_commit_sha", sa.String(length=80), nullable=True),
}


def upgrade() -> None:
    bind = op.get_bind()
    if "model_versions" not in inspect(bind).get_table_names():
        return

    existing = {column["name"] for column in inspect(bind).get_columns("model_versions")}
    for name, column in ARTIFACT_COLUMNS.items():
        if name not in existing:
            op.add_column("model_versions", column)


def downgrade() -> None:
    bind = op.get_bind()
    if "model_versions" not in inspect(bind).get_table_names():
        return

    existing = {column["name"] for column in inspect(bind).get_columns("model_versions")}
    for name in reversed(tuple(ARTIFACT_COLUMNS)):
        if name in existing:
            op.drop_column("model_versions", name)
