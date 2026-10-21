import re
from typing import Annotated

import structlog
from fastapi import Depends, Header, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import ANONYMOUS, Principal, TokenError, decode_access_token
from app.core.config import get_settings
from app.core.idempotency import IdempotencyStore
from app.core.logging import get_logger
from app.core.rate_limit import RateLimiter
from app.db.base import get_session
from app.llm.ollama import OllamaClient

logger = get_logger(__name__)


def get_ollama(request: Request) -> OllamaClient:
    """The shared model client on app.state (created at startup).

    Always embeds through local Ollama; `chat()` / `chat_stream()` go to the
    LLM_PROVIDER model (an `AnthropicChatClient` is an `OllamaClient` subclass).
    """
    return request.app.state.ollama


def client_identity(request: Request) -> str:
    """Who a request counts against for rate limiting.

    Authenticated: the token's subject, so a client's budget follows it
    across IP addresses and can't be dodged by rotating them (or shared by
    everyone behind one NAT). Unauthenticated (only /auth/token, or auth
    switched off): the client IP as seen by the server. Never a client-sent
    header like X-Forwarded-For, which anyone could change to reset their
    limit.
    """
    principal: Principal | None = getattr(request.state, "principal", None)
    if principal is not None and principal.authenticated:
        return f"sub:{principal.subject}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


_bearer = HTTPBearer(auto_error=False)


def _unauthorized(error: str | None = None) -> HTTPException:
    challenge = "Bearer" if error is None else f'Bearer error="{error}"'
    return HTTPException(
        status_code=401,
        detail="Missing or invalid access token.",
        headers={"WWW-Authenticate": challenge},
    )


async def authenticate(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    """Resolve the caller from the bearer token, or reject with 401."""
    settings = get_settings()
    if not settings.auth_enabled:
        request.state.principal = ANONYMOUS
        return ANONYMOUS

    if credentials is None:
        raise _unauthorized()
    try:
        principal = decode_access_token(settings, credentials.credentials)
    except TokenError as exc:
        # The reason goes to the log; the client only learns "invalid".
        logger.info("token rejected", reason=exc.reason)
        raise _unauthorized("invalid_token") from exc

    request.state.principal = principal
    # Every log line for the rest of this request now names the caller.
    structlog.contextvars.bind_contextvars(subject=principal.subject)
    return principal


PrincipalDep = Annotated[Principal, Depends(authenticate)]


def require_scope(scope: str):
    """Route dependency: the caller's token must carry `scope`, else 403."""

    async def check(principal: PrincipalDep) -> Principal:
        if not principal.has_scope(scope):
            logger.info("scope denied", subject=principal.subject, required_scope=scope)
            raise HTTPException(
                status_code=403, detail=f"This action requires the '{scope}' scope."
            )
        return principal

    return check


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


# Letters, digits and . _ : - only, 1-128 chars: safe to log and to store.
_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def idempotency_key(
    idempotency_key: Annotated[str | None, Header()] = None,
) -> str | None:
    """The request's `Idempotency-Key` header, validated, or None if absent."""
    if idempotency_key is None:
        return None
    if not _IDEMPOTENCY_KEY.match(idempotency_key):
        raise HTTPException(
            status_code=400,
            detail="Idempotency-Key must be 1-128 characters from A-Z a-z 0-9 . _ : -",
        )
    return idempotency_key


def get_idempotency_store(request: Request) -> IdempotencyStore | None:
    """The shared store, or None when IDEMPOTENCY_ENABLED=false."""
    return getattr(request.app.state, "idempotency", None)


SessionDep = Annotated[AsyncSession, Depends(get_session)]
IdempotencyKeyDep = Annotated[str | None, Depends(idempotency_key)]
IdempotencyStoreDep = Annotated[IdempotencyStore | None, Depends(get_idempotency_store)]
OllamaDep = Annotated[OllamaClient, Depends(get_ollama)]
