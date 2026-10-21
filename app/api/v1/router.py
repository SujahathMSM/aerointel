import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
)
from fastapi.responses import StreamingResponse

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
from app.core.sse import format_event
from app.schemas.documents import DocumentResponse
from app.schemas.ingest import IngestRequest, IngestResponse
from app.schemas.query import (
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    VerificationOut,
)
from app.schemas.search import SearchRequest, SearchResponse
from app.services.documents import ingest_pdf
from app.services.generation import (
    answer_question,
    prepare_answer,
    refusal_message,
    stream_answer,
    verify_prepared,
)
from app.services.ingestion import ingest_chunk
from app.services.pdf_text import PdfError
from app.services.pipeline import retrieve
from app.services.verification import Verification

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


def _verification_out(
    v: Verification | None, blocked: bool = False
) -> VerificationOut | None:
    if v is None:
        return None
    return VerificationOut(
        ok=v.ok,
        checked=v.checked,
        cited=v.cited,
        unsupported=v.unsupported,
        unit_mismatch=v.unit_mismatch,
        bad_citations=v.bad_citations,
        found_in_uncited=v.found_in_uncited,
        blocked=blocked,
    )


def _refusal_text(reason, unknown_terms, verification) -> str:
    figures = (
        [*verification.unsupported, *verification.unit_mismatch] if verification else []
    )
    return refusal_message(reason, unknown_terms, figures, get_settings().max_distance)


def _model_name(llm) -> str | None:
    """The answering model's name, for the response and the logs."""
    return getattr(llm, "model_name", None)


def _to_retrieved(chunk, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk.id,
        distance=distance,
        source=chunk.metadata_.get("source"),
        content=chunk.content,
        document_id=getattr(chunk, "document_id", None),
        page=getattr(chunk, "page", None),
        page_label=getattr(chunk, "page_label", None),
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_scope(SCOPE_QUERY))],
)
async def query(body: QueryRequest, session: SessionDep, ollama: OllamaDep):
    """Grounded answer with [n] citations, or a structural refusal."""
    result = await answer_question(session, ollama, body.question, body.top_k)

    if result.refused:
        return QueryResponse(
            question=result.question,
            refused=True,
            refusal_reason=result.refusal_reason,
            refusal_message=_refusal_text(
                result.refusal_reason, result.unknown_terms, result.verification
            ),
            unknown_terms=result.unknown_terms,
            chunks=[_to_retrieved(c, d) for c, d in result.chunks],
            prompt_version=result.prompt_version,
            verification=_verification_out(result.verification, blocked=True)
            if result.verification
            else None,
            answer_check=result.answer_check,
        )

    logger.info(
        "query answered",
        model=_model_name(ollama),
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
        model=_model_name(ollama),
        unknown_terms=result.unknown_terms,
        verification=_verification_out(result.verification),
        answer_check=result.answer_check,
    )


@router.post(
    "/query/stream",
    dependencies=[Depends(require_scope(SCOPE_QUERY))],
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def query_stream(body: QueryRequest, session: SessionDep, ollama: OllamaDep):
    """Same grounded answer as /v1/query, delivered as Server-Sent Events.

    Retrieval and the refusal gate run *before* the response starts, so a
    bad request, missing scope, rate limit or retrieval failure is an ordinary
    HTTP error. After that the stream is:

    - `meta`  -- question, refused, refusal_reason, refusal_message,
                 unknown_terms, answer_check, chunks (the sources),
                 prompt_version, model (who writes the answer). Always first.
    - `token` -- `{"text": "..."}`, repeated; concatenate to build the answer.
    - `verification` -- only when the answer contained figures or citations to
                 check: the same object as `/v1/query`'s `verification`. The text
                 is already sent, so with VERIFY_MODE=block `blocked: true` means
                 "discard what you received". Absent when there was nothing to check.
    - `done`  -- `{"refused": bool, "answer_chars": n}`. Always last on success.
    - `error` -- `{"detail": "..."}` if generation fails after the stream began
                 (the status line was already sent); no `done` follows.

    A refusal is `meta` (refused=true) then `done`: the model is never asked to
    answer. (With ANSWER_CHECK=refuse it may have been asked the yes/no check,
    which is what refused it. A figure-check withhold is decided after
    generation and so cannot happen on the stream: see `verification`.)
    """
    settings = get_settings()
    prepared = await prepare_answer(session, ollama, body.question, body.top_k)

    meta = QueryResponse(
        question=prepared.question,
        refused=prepared.refused,
        refusal_reason=prepared.refusal_reason,
        refusal_message=(
            _refusal_text(prepared.refusal_reason, prepared.unknown_terms, None)
            if prepared.refused
            else None
        ),
        unknown_terms=prepared.unknown_terms,
        chunks=[_to_retrieved(c, d) for c, d in prepared.chunks],
        prompt_version=prepared.prompt_version,
        model=None if prepared.refused else _model_name(ollama),
        answer_check=prepared.answer_check,
    ).model_dump(exclude={"answer", "verification"})

    async def events() -> AsyncIterator[str]:
        started = time.monotonic()
        chars = 0
        outcome = "error"
        try:
            yield format_event("meta", meta)
            if prepared.refused:
                outcome = "refused"
                yield format_event("done", {"refused": True, "answer_chars": 0})
                return
            pieces: list[str] = []
            async with aclosing(stream_answer(ollama, prepared)) as tokens:
                async for piece in tokens:
                    chars += len(piece)
                    pieces.append(piece)
                    yield format_event("token", {"text": piece})
            outcome = "completed"
            # The text is already on the wire, so a failed check cannot withdraw
            # it; the client is told, and in "block" mode must discard it.
            verification = verify_prepared(prepared, "".join(pieces))
            if verification and (verification.checked or verification.bad_citations):
                blocked = not verification.ok and settings.verify_mode == "block"
                out = _verification_out(verification, blocked=blocked)
                assert out is not None
                yield format_event("verification", out.model_dump())
            yield format_event("done", {"refused": False, "answer_chars": chars})
        except (asyncio.CancelledError, GeneratorExit):
            outcome = "client_disconnected"
            raise
        except Exception:
            # The 200 is already on the wire: report in-band, keep internals
            # (hostnames, model errors) in the logs rather than the stream.
            logger.exception("query stream failed")
            yield format_event("error", {"detail": "Answer generation failed."})
        finally:
            logger.info(
                "query stream finished",
                outcome=outcome,
                answer_chars=chars,
                prompt_version=prepared.prompt_version,
                duration_ms=round((time.monotonic() - started) * 1000),
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tell nginx not to buffer the stream
        },
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
    "/documents",
    response_model=DocumentResponse,
    dependencies=[Depends(require_scope(SCOPE_INGEST))],
)
async def upload_document(
    file: Annotated[UploadFile, File(description="The PDF to ingest.")],
    session: SessionDep,
    ollama: OllamaDep,
    title: Annotated[str | None, Form(max_length=300)] = None,
):
    """Upload a PDF: it is read, cut into page-bound chunks, embedded and stored.

    Answers cite the document and page (both the PDF page and the number
    printed on it). Uploading the same file again changes nothing and returns
    the existing document with `duplicate: true`.

    The request stays open until embedding finishes, which can take minutes
    for a long manual; for those use `python scripts/ingest_pdfs.py` until
    background ingestion arrives.
    """
    limit = get_settings().max_upload_mb * 1024 * 1024
    data = bytearray()
    while piece := await file.read(1024 * 1024):
        data += piece
        if len(data) > limit:
            raise HTTPException(
                status_code=413,
                detail=f"File is larger than the {get_settings().max_upload_mb} MB limit.",
            )
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    try:
        result = await ingest_pdf(
            session, ollama, bytes(data), file.filename or "document.pdf", title
        )
    except PdfError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return DocumentResponse(**vars(result))


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
