from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentChunk
from app.llm.ollama import OllamaClient


async def vector_search(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[tuple]:
    """Rank chunks by cosine distance to a precomputed query embedding.

    Takes the embedding (not the text) so callers that already embedded —
    e.g. hybrid search, which needs the same vector for the distance
    backfill — don't pay for a second Ollama round-trip.
    """
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(DocumentChunk, distance.label("distance"))
        .where(DocumentChunk.embedding.is_not(None))
        .order_by(distance)
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).all()
    return [(chunk, float(row_distance)) for chunk, row_distance in rows]


async def search_chunks(
    session: AsyncSession,
    ollama: OllamaClient,
    query: str,
    top_k: int = 5,
    query_embedding: list[float] | None = None,
) -> list[tuple]:
    """Embed the query, return (chunk, distance) closest-first via pgvector.

    `query_embedding`: the query's vector if the caller already has it.
    """
    if query_embedding is None:
        (query_embedding,) = await ollama.embed([query])
    return await vector_search(session, query_embedding, top_k)
