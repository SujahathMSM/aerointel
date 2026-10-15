from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


class Document(Base):
    """One source file (a PDF) that chunks were cut from.

    `content_hash` is the SHA-256 of the file's bytes. It is unique, so the
    same file can only be ingested once however many times it is uploaded
    (content-hash deduplication).
    """

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_documents_content_hash"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Pages whose text came from OCR rather than the PDF's own text layer.
    ocr_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DocumentChunk(Base):
    """One retrievable piece of a document plus its embedding."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        # Mirrors alembic/versions/0002: same GIN index, declared here too
        # so Base.metadata.create_all() (used by test fixtures) produces
        # the identical schema Alembic produces against a real database.
        Index("idx_document_chunks_tsv", "tsv", postgresql_using="gin"),
        Index("idx_document_chunks_document_id", "document_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # "metadata" is reserved by SQLAlchemy, so the attribute is metadata_
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(get_settings().embedding_dimensions), nullable=True
    )
    # Which model produced `embedding` (Phase 1 embedding-version tracking).
    # Nullable: pre-Phase-1 rows and rows pending scripts/backfill_embeddings.py
    # carry NULL until backfilled. Stamped on every new write by ingest_chunk().
    embedding_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Where the chunk came from. All NULL for chunks ingested as plain text
    # through POST /v1/ingest; set for chunks cut from an uploaded PDF.
    document_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), nullable=True
    )
    # Position of the chunk within its document (0-based, reading order).
    chunk_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 1-based page of the PDF file itself (what a PDF viewer calls page N).
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # The page number printed on the page, e.g. "3-3" or "xii", when one could
    # be read from it. Maintenance manuals cite these, not file page numbers.
    page_label: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Full-text search vector, kept in sync with `content` by Postgres itself
    # (GENERATED ALWAYS AS ... STORED). Read-only from the ORM's side — never
    # assign to this column, Postgres computes it. Declared here (not just in
    # the migration) so SQLAlchemy's own Base.metadata.create_all() — what the
    # integration-test fixtures use — creates the identical column a real
    # `alembic upgrade head` does. See app/services/hybrid.py:keyword_search.
    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=True,
    )
