"""The retrieval pipeline: search (vector or hybrid) -> rerank -> top_k.

One function the API (step 5) and the eval harness both call, so what is
measured is exactly what is served. Every stage is individually
feature-flagged; with everything off this is the Phase 0 baseline.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.llm.ollama import OllamaClient
from app.services.hybrid import hybrid_search
from app.services.reranking import NoReranker, Reranker, get_reranker, rerank
from app.services.retrieval import search_chunks


async def retrieve(
    session: AsyncSession,
    ollama: OllamaClient,
    query: str,
    top_k: int | None = None,
    *,
    hybrid: bool | None = None,
    reranker: Reranker | None = None,
) -> list[tuple]:
    """Return (chunk, distance) pairs, best first.

    hybrid:   None -> use settings.hybrid_enabled; explicit True/False
              overrides (the eval harness A/B tests this way).
    reranker: None -> use the configured reranker from settings.
    """
    settings = get_settings()
    top_k = top_k or settings.top_k
    use_hybrid = settings.hybrid_enabled if hybrid is None else hybrid
    active_reranker = reranker or get_reranker()

    # When reranking, fetch extra candidates: the whole point is that the
    # right answer may sit at rank 7 where the first-stage ranker put it,
    # and only the cross-encoder can see it belongs at rank 1.
    reranking_active = not isinstance(active_reranker, NoReranker)
    candidate_k = max(top_k, settings.rerank_candidates) if reranking_active else top_k

    if use_hybrid:
        candidates = await hybrid_search(session, ollama, query, candidate_k)
    else:
        candidates = await search_chunks(session, ollama, query, candidate_k)

    return rerank(query, candidates, active_reranker, top_k)
