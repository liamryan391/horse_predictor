"""Initial SQL data platform schema.

Revision ID: 20260731_0001
Revises:
Create Date: 2026-07-31
"""

from __future__ import annotations

from alembic import op

from schema import METADATA

revision = "20260731_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    METADATA.create_all(bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    for table in reversed(METADATA.sorted_tables):
        table.drop(bind, checkfirst=True)
