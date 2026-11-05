"""add document revision, document key and superseded flag

Maintenance must be done to current data, so every answer names the revision
of the document it came from. `documents` gains the revision ("AC 43.13-1B CHG
1", parsed from the running headers or given at upload), its date, a
`document_key` that stays the same across revisions ("AC 43.13-1B", so a
newer stored revision can be noticed) and a manual `superseded` flag. Each
chunk's own page issue lives in its metadata (a change-page document mixes
issues), so `document_chunks` needs no new column.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-05

"""
from typing import Sequence

from alembic import context, op
import sqlalchemy as sa

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Idempotent like 0004: the integration tests' create_all() can add these
    # columns to the development database before this migration runs.
    inspector = None if context.is_offline_mode() else sa.inspect(op.get_bind())
    existing = (
        set()
        if inspector is None
        else {c["name"] for c in inspector.get_columns("documents")}
    )
    for column in [
        sa.Column("revision", sa.Text(), nullable=True),
        sa.Column("revision_date", sa.Date(), nullable=True),
        sa.Column("document_key", sa.Text(), nullable=True),
        sa.Column(
            "superseded", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
    ]:
        if column.name not in existing:
            op.add_column("documents", column)

    indexes = (
        set()
        if inspector is None
        else {ix["name"] for ix in inspector.get_indexes("documents")}
    )
    if "idx_documents_document_key" not in indexes:
        op.create_index("idx_documents_document_key", "documents", ["document_key"])


def downgrade() -> None:
    op.drop_index("idx_documents_document_key", table_name="documents")
    op.drop_column("documents", "superseded")
    op.drop_column("documents", "document_key")
    op.drop_column("documents", "revision_date")
    op.drop_column("documents", "revision")
