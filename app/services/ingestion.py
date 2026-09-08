from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import DocumentChunk
from app.llm.ollama import OllamaClient


async def ingest_chunk(
    session: AsyncSession,
    ollama: OllamaClient,
    content: str,
    metadata: dict | None = None,
) -> DocumentChunk:
    """Embed one chunk of text and store it with its vector.

    Stamps `embedding_model` with the model that made this vector (Phase 1
    embedding-version tracking), so a future model swap can tell old and
    new rows apart instead of silently mixing incompatible vectors.
    """
    (embedding,) = await ollama.embed([content])

    chunk = DocumentChunk(
        content=content,
        metadata_=metadata or {},
        embedding=embedding,
        embedding_model=get_settings().embedding_model,
    )
    session.add(chunk)
    await session.commit()
    await session.refresh(chunk)
    return chunk
