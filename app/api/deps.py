from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.rate_limit import RateLimiter
from app.db.base import get_session
from app.llm.ollama import OllamaClient

logger = get_logger(__name__)


def get_ollama(request: Request) -> OllamaClient:
    """The shared OllamaClient lives on app.state (created at startup)."""
    return request.app.state.ollama


def client_identity(request: Request) -> str:
    """Who a request counts against for rate limiting.

    For now: the client IP as seen by the server. There is no auth yet, so
    this is the only identity the API can't be talked out of — trusting a
    client-sent header (an API key nobody validates, X-Forwarded-For) would
    let anyone reset their limit by changing the header. Once JWT auth
    lands later in Phase 2, this becomes the token's subject.
    """
    return f"ip:{request.client.host if request.client else 'unknown'}"


async def enforce_rate_limit(request: Request, response: Response) -> None:
    """Spend one token for this client, or reject with 429."""
    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:  # RATE_LIMIT_ENABLED=false
        return

    result = await limiter.hit(client_identity(request))
    headers = {
        "X-RateLimit-Limit": str(result.limit),
        "X-RateLimit-Remaining": str(result.remaining),
    }

    if result.allowed:
        response.headers.update(headers)
        return

    if result.degraded:
        # Only reachable with RATE_LIMIT_FAIL_OPEN=false and Redis down.
        raise HTTPException(status_code=503, detail="rate limiter unavailable")

    logger.info(
        "rate limited",
        identity=client_identity(request),
        retry_after_seconds=result.retry_after_seconds,
    )
    raise HTTPException(
        status_code=429,
        detail="Rate limit exceeded. Retry after the number of seconds "
        "in the Retry-After header.",
        headers={**headers, "Retry-After": str(result.retry_after_seconds)},
    )


SessionDep = Annotated[AsyncSession, Depends(get_session)]
OllamaDep = Annotated[OllamaClient, Depends(get_ollama)]
