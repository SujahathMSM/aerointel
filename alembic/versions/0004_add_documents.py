"""add documents table and per-chunk page provenance

PDF ingestion needs to say where an answer came from. `documents` holds one row
per uploaded file (unique SHA-256 of its bytes, so the same file is only ever
ingested once). `document_chunks` gains the owning document, the chunk's
position, and the page it came from (both the PDF page and the number printed
on the page). All new chunk columns are nullable: chunks ingested as plain text
through POST /v1/ingest have no document and keep working unchanged.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-14

"""
from typing import Sequence, Union

from alembic import context, op
import sqlalchemy as sa

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Every step first checks whether it is already done. The integration tests
    # build tables with Base.metadata.create_all() in the development database,
    # which can create `documents` before this migration has run; a plain
    # create_table would then fail with "relation already exists".
    # Offline mode (`alembic upgrade --sql`) has no database to inspect: emit
    # the plain statements for a database that is at revision 0003.
    inspector = None if context.is_offline_mode() else sa.inspect(op.get_bind())

    if inspector is None or not inspector.has_table("documents"):
        _create_documents()

    existing = (
        set()
        if inspector is None
        else {c["name"] for c in inspector.get_columns("document_chunks")}
    )
    for name, type_ in [
        ("document_id", sa.BigInteger()),
        ("chunk_index", sa.Integer()),
        ("page", sa.Integer()),
        ("page_label", sa.Text()),
    ]:
        if name not in existing:
            op.add_column("document_chunks", sa.Column(name, type_, nullable=True))

    foreign_keys = (
        set()
        if inspector is None
        else {fk["name"] for fk in inspector.get_foreign_keys("document_chunks")}
    )
    if "fk_document_chunks_document_id" not in foreign_keys:
        op.create_foreign_key(
            "fk_document_chunks_document_id",
            "document_chunks",
            "documents",
            ["document_id"],
            ["id"],
            ondelete="CASCADE",
        )
    indexes = (
        set()
        if inspector is None
        else {ix["name"] for ix in inspector.get_indexes("document_chunks")}
    )
    if "idx_document_chunks_document_id" not in indexes:
        op.create_index(
            "idx_document_chunks_document_id", "document_chunks", ["document_id"]
        )


def _create_documents() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("ocr_pages", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("content_hash", name="uq_documents_content_hash"),
    )


def downgrade() -> None:
    op.drop_index("idx_document_chunks_document_id", table_name="document_chunks")
    op.drop_constraint(
        "fk_document_chunks_document_id", "document_chunks", type_="foreignkey"
    )
    op.drop_column("document_chunks", "page_label")
    op.drop_column("document_chunks", "page")
    op.drop_column("document_chunks", "chunk_index")
    op.drop_column("document_chunks", "document_id")
    op.drop_table("documents")
