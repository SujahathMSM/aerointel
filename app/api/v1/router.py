from fastapi import APIRouter, Depends, HTTPException, Response

from app.api.deps import (
    IdempotencyKeyDep,
    IdempotencyStoreDep,
    OllamaDep,
    PrincipalDep,
    SessionDep,
    authenticate,
    enforce_rate_limit,
    require_scope,
)
from app.core.auth import SCOPE_INGEST, SCOPE_QUERY
from app.core.config import get_settings
from app.core.idempotency import (
    IdempotencyInProgress,
    IdempotencyMismatch,
    IdempotencyUnavailable,
    fingerprint,
    run_once,
)
from app.core.logging import get_logger
from app.schemas.ingest import IngestRequest, IngestResponse
from app.schemas.query import QueryRequest, QueryResponse, RetrievedChunk
from app.schemas.search import SearchRequest, SearchResponse
from app.services.generation import NO_CONTEXT_MESSAGE, answer_question
from app.services.ingestion import ingest_chunk
from app.services.pipeline import retrieve

# Every /v1 route first proves who is calling (401 if not), then spends a
# rate-limit token from that caller's bucket; each route then checks its
# scope. Order matters: authenticate runs first so the bucket is keyed by the
# token's subject. Health checks live on a separate router on purpose: a load
# balancer probing /health must never be throttled or asked for credentials.
router = APIRouter(
    prefix="/v1",
    tags=["v1"],
    dependencies=[Depends(authenticate), Depends(enforce_rate_limit)],
)
logger = get_logger(__name__)


def _to_retrieved(chunk, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk.id,
        distance=distance,
        source=chunk.metadata_.get("source"),
        content=chunk.content,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_scope(SCOPE_QUERY))],
)
async def query(body: QueryRequest, session: SessionDep, ollama: OllamaDep):
    """Grounded answer with [n] citations, or a structural refusal."""
    result = await answer_question(session, ollama, body.question, body.top_k)

    settings = get_settings()

    if result.refused:
        return QueryResponse(
            question=result.question,
            refused=True,
            refusal_message=NO_CONTEXT_MESSAGE.format(ceiling=settings.max_distance),
            chunks=[_to_retrieved(c, d) for c, d in result.chunks],
        )

    logger.info(
        "query answered",
        chunks_used=len(result.chunks),
        answer_chars=len(result.answer or ""),
        prompt_version=result.prompt_version,
    )
    return QueryResponse(
        question=result.question,
        refused=False,
        answer=result.answer,
        chunks=[_to_retrieved(c, d) for c, d in result.chunks],
        prompt_version=result.prompt_version,
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    dependencies=[Depends(require_scope(SCOPE_QUERY))],
)
async def search(body: SearchRequest, session: SessionDep, ollama: OllamaDep):
    """Search only — no generation, no LLM call.

    Same retrieval pipeline as /v1/query (hybrid + reranking, both
    feature-flagged in settings), so what this endpoint shows you is
    exactly what /v1/query's grounding actually saw.
    """
    settings = get_settings()
    results = await retrieve(session, ollama, body.query, body.top_k or settings.top_k)
    return SearchResponse(
        query=body.query,
        results=[_to_retrieved(chunk, distance) for chunk, distance in results],
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(require_scope(SCOPE_INGEST))],
)
async def ingest(
    body: IngestRequest,
    response: Response,
    session: SessionDep,
    ollama: OllamaDep,
    principal: PrincipalDep,
    key: IdempotencyKeyDep,
    store: IdempotencyStoreDep,
):
    """Embed and store one chunk.

    Send an `Idempotency-Key` header to make retries safe: the first request
    with a given key stores the chunk; a retry with the same key and body gets
    the same response back (marked `Idempotent-Replay: true`) and stores
    nothing. The same key with a different body is rejected (422), and a retry
    that arrives while the first is still running gets 409.
    """
    settings = get_settings()

    async def store_chunk() -> dict:
        chunk = await ingest_chunk(session, ollama, body.content, body.metadata)
        logger.info("chunk ingested", chunk_id=chunk.id)
        return IngestResponse(
            id=chunk.id, dimensions=settings.embedding_dimensions
        ).model_dump()

    if key is None or store is None:  # no key sent, or idempotency switched off
        return IngestResponse.model_validate(await store_chunk())

    try:
        result = await run_once(
            store,
            principal.subject,
            key,
            fingerprint(body.model_dump()),
            store_chunk,
        )
    except IdempotencyMismatch:
        raise HTTPException(
            status_code=422,
            detail="This Idempotency-Key was already used with a different "
            "request body. Use a new key for a new request.",
        ) from None
    except IdempotencyInProgress:
        raise HTTPException(
            status_code=409,
            detail="A request with this Idempotency-Key is still being "
            "processed. Retry shortly.",
            headers={"Retry-After": "1"},
        ) from None
    except IdempotencyUnavailable as exc:
        # Refuse rather than run unprotected: a key means "exactly once".
        logger.error("idempotency store unavailable", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="Idempotency protection is unavailable. Retry shortly.",
            headers={"Retry-After": "5"},
        ) from exc

    if result.replayed:
        response.headers["Idempotent-Replay"] = "true"
    return IngestResponse.model_validate(result.body)
