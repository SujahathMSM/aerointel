from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentChunk
from app.llm.ollama import OllamaClient


async def ingest_chunk(
    session: AsyncSession,
    ollama: OllamaClient,
    content: str,
    metadata: dict | None = None,
) -> DocumentChunk:
    """Embed one chunk of text and store it with its vector."""
    (embedding,) = await ollama.embed([content])

    chunk = DocumentChunk(
        content=content,
        metadata_=metadata or {},
        embedding=embedding,
    )
    session.add(chunk)
    await session.commit()
    await session.refresh(chunk)
    return chunk
