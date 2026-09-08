"""add tsvector column + GIN index for keyword (BM25-style) search

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-01

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Generated column: Postgres keeps it in sync with content automatically.
    # 'english' config handles stemming ("brakes" matches "brake") and
    # stopwords. STORED = computed on write, cheap to query.
    op.execute(
        """
        ALTER TABLE document_chunks
        ADD COLUMN tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """
    )
    # GIN index: makes @@ (match) lookups fast instead of a full scan.
    op.execute(
        "CREATE INDEX idx_document_chunks_tsv "
        "ON document_chunks USING GIN (tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_tsv")
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS tsv")