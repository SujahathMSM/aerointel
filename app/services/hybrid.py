"""Hybrid retrieval: keyword (tsvector) + vector search, fused via RRF.

Why hybrid: embeddings match by meaning but miss exact terms ("fuse plug",
"2.5 bar"); keyword search hits exact terms but misses paraphrases. Each
covers the other's blind spot. Reciprocal Rank Fusion merges the two
rankings without needing score normalization (ts_rank scores and cosine
distances live on incomparable scales — ranks don't).

The keyword half is Postgres full-text search ranked by ts_rank. That is not
BM25: ts_rank does not weight rare words higher (no IDF) and by default does
not correct for chunk length. RRF only uses the order of each list, which
softens the difference.
"""

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import DocumentChunk
from app.llm.ollama import OllamaClient
from app.services.answerability import question_terms
from app.services.retrieval import vector_search


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion over several id rankings.

    score(id) = sum over rankings of 1 / (k + rank), rank 1-based.
    k=60 dampens the influence of top ranks (the standard constant from
    the original RRF paper). Returns ids sorted by fused score, best first.
    Ties keep insertion order (Python's sort is stable).
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def keyword_query_text(query: str, mode: str = "all") -> str | None:
    """The text handed to websearch_to_tsquery; None = nothing to search for.

    "all": the question as typed. websearch_to_tsquery joins its words with AND,
    so only a chunk containing every word matches.
    "any": the question's distinctive words (the same ones the unknown-word
    check uses: no stopwords, no plain numbers) joined with "or", so a chunk
    matching any of them is a candidate and ts_rank prefers the chunks that
    match more of them.
    """
    if mode == "all":
        return query
    if mode != "any":
        raise ValueError(f"unknown keyword query mode: {mode!r}")
    terms = question_terms(query)
    return " or ".join(terms) if terms else None


async def keyword_search(
    session: AsyncSession,
    query: str,
    top_k: int = 5,
    mode: str | None = None,
) -> list[tuple]:
    """Full-text match against the generated tsv column.

    websearch_to_tsquery parses plain user text (quotes, OR, minus-terms)
    without needing special syntax. Returns (chunk, ts_rank) ordered by
    rank score descending. Chunks with no keyword overlap are simply
    absent — that's the signal RRF needs.

    `mode`: "all" or "any" (see keyword_query_text); None = KEYWORD_QUERY.
    """
    q = keyword_query_text(query, mode or get_settings().keyword_query)
    if q is None:
        return []
    id_rank = text(
        """
        SELECT id, ts_rank(tsv, websearch_to_tsquery('english', :q)) AS rank
        FROM document_chunks
        WHERE tsv @@ websearch_to_tsquery('english', :q)
        ORDER BY rank DESC
        LIMIT :k
        """
    )
    rows = (await session.execute(id_rank, {"q": q, "k": top_k})).all()

    if not rows:
        return []

    id_to_rank = {row_id: float(rank) for row_id, rank in rows}
    chunks = list(
        (
            await session.execute(
                select(DocumentChunk).where(DocumentChunk.id.in_(id_to_rank))
            )
        )
        .scalars()
        .all()
    )
    # Preserve the ts_rank ordering from the first query.
    chunks.sort(key=lambda c: id_to_rank[c.id], reverse=True)
    return [(chunk, id_to_rank[chunk.id]) for chunk in chunks]


async def hybrid_search(
    session: AsyncSession,
    ollama: OllamaClient,
    query: str,
    top_k: int = 5,
    keyword_query: str | None = None,
) -> list[tuple]:
    """Embed once, search both ways, fuse with RRF, backfill distances.

    Returns (chunk, cosine_distance) in fused order — same shape as
    search_chunks, so the distance gate in generation.py works unchanged.
    Keyword-only hits get their distance computed in one batched query so
    the gate can still judge them.
    """
    settings = get_settings()
    (query_embedding,) = await ollama.embed([query])

    vector_results = await vector_search(session, query_embedding, top_k)
    keyword_results = await keyword_search(session, query, top_k, keyword_query)

    fused_ids = rrf_fuse(
        [
            [chunk.id for chunk, _ in vector_results],
            [chunk.id for chunk, _ in keyword_results],
        ],
        k=settings.rrf_k,
    )[:top_k]

    if not fused_ids:
        return []

    # One batched query for every fused chunk: entity + cosine distance.
    distance_expr = DocumentChunk.embedding.cosine_distance(query_embedding)
    stmt = select(DocumentChunk, distance_expr.label("distance")).where(
        DocumentChunk.id.in_(fused_ids)
    )
    by_id = {
        chunk.id: (chunk, float(dist))
        for chunk, dist in (await session.execute(stmt)).all()
    }

    # Emit in fused order; a chunk deleted mid-flight is skipped, not fatal.
    return [by_id[cid] for cid in fused_ids if cid in by_id]
