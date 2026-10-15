"""Widen what the model reads: add the chunks that follow each retrieved one.

Chunks are cut at sentence ends and never cross a page, so an answer that runs
on ("c. ... d. ... e. ...") is often split over two or three consecutive
chunks while only one of them matched the question. Pulling in the chunk after
each hit gives the model the rest of the passage without retrieving anything
extra by similarity (which finds look-alikes, not continuations).

Only chunks cut from a PDF have a position (`document_id`, `chunk_index`);
anything else is passed through untouched.
"""

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentChunk

MAX_NEIGHBOURS = 5


def _position(chunk) -> tuple[int, int] | None:
    document_id = getattr(chunk, "document_id", None)
    index = getattr(chunk, "chunk_index", None)
    if document_id is None or index is None:
        return None
    return document_id, index


def plan_neighbours(
    results: list[tuple], neighbours: int, max_chunks: int
) -> dict[int, list[tuple[int, int]]]:
    """Decide which following chunks to fetch: {hit chunk id: [(document, index)]}.

    Pure function. The first following chunk of *every* hit is planned before
    the second following chunk of any hit, so the budget is spread over the hits
    instead of the best hit taking all of it. A chunk that is already a hit, or
    already planned for another hit, is not planned twice. The hits themselves
    always stay: `max_chunks` only limits how many neighbours are added.
    """
    neighbours = max(0, min(neighbours, MAX_NEIGHBOURS))
    budget = max(0, max_chunks - len(results))
    taken = {pos for chunk, _ in results if (pos := _position(chunk))}
    plan: dict[int, list[tuple[int, int]]] = {}
    for step in range(1, neighbours + 1):
        for chunk, _ in results:
            pos = _position(chunk)
            if pos is None:
                continue
            if budget <= 0:
                return plan
            wanted = (pos[0], pos[1] + step)
            if wanted in taken:
                continue
            taken.add(wanted)
            plan.setdefault(chunk.id, []).append(wanted)
            budget -= 1
    return plan


async def expand_with_neighbours(
    session: AsyncSession,
    results: list[tuple],
    neighbours: int,
    max_chunks: int,
) -> list[tuple]:
    """Return `results` with each hit followed by its next chunk(s).

    Order: every hit stays in its retrieved position and its neighbours come
    right after it, so citation [1] is still the best match and a continuation
    reads straight after the passage it continues. A neighbour carries its
    hit's distance (it was not itself scored) so it passes the same gate.
    A neighbour that does not exist (hit was the last chunk) is skipped.
    """
    plan = plan_neighbours(results, neighbours, max_chunks)
    if not plan:
        return list(results)

    keys = [key for keys in plan.values() for key in keys]
    rows = (
        (
            await session.execute(
                select(DocumentChunk).where(
                    tuple_(DocumentChunk.document_id, DocumentChunk.chunk_index).in_(
                        keys
                    )
                )
            )
        )
        .scalars()
        .all()
    )
    found = {(row.document_id, row.chunk_index): row for row in rows}

    expanded: list[tuple] = []
    for chunk, distance in results:
        expanded.append((chunk, distance))
        for key in plan.get(chunk.id, []):
            neighbour = found.get(key)
            if neighbour is not None:
                expanded.append((neighbour, distance))
    return expanded
