from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Computed, Index, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


class DocumentChunk(Base):
    """One retrievable piece of a document plus its embedding."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        # Mirrors alembic/versions/0002: same GIN index, declared here too
        # so Base.metadata.create_all() (used by test fixtures) produces
        # the identical schema Alembic produces against a real database.
        Index("idx_document_chunks_tsv", "tsv", postgresql_using="gin"),
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
