from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentChunk
from app.llm.ollama import OllamaClient


async def search_chunks(
    session: AsyncSession,
    ollama: OllamaClient,
    query: str,
    top_k: int = 5,
) -> list[tuple]:
    """Embed the query, return (chunk, distance) closest-first via pgvector."""
    (query_embedding,) = await ollama.embed([query])

    distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(DocumentChunk, distance.label("distance"))
        .where(DocumentChunk.embedding.is_not(None))
        .order_by(distance)
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).all()
    return [(chunk, float(row_distance)) for chunk, row_distance in rows]
