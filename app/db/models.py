from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


class DocumentChunk(Base):
    """One retrievable piece of a document plus its embedding."""

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # "metadata" is reserved by SQLAlchemy, so the attribute is metadata_
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(get_settings().embedding_dimensions), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
