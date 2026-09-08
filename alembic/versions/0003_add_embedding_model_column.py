"""add embedding_model column (embedding-version tracking)

Different embedding models produce incompatible vectors. This column
records which model produced each row's `embedding`, so a future model
swap can be rolled out safely: backfill new vectors under the new model
name via scripts/backfill_embeddings.py, then flip the read path, instead
of silently mixing old and new vectors in the same searches.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Nullable, no default: existing rows stay NULL until
    # scripts/backfill_embeddings.py stamps them explicitly. A silent
    # default here would claim untracked old rows were made by whatever
    # model happens to be configured today, which may not be true.
    op.add_column(
        "document_chunks",
        sa.Column("embedding_model", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("document_chunks", "embedding_model")
