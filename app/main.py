from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.auth import router as auth_router
from app.api.health import router as health_router
from app.api.v1.router import router as v1_router
from app.core.auth import validate_auth_settings
from app.core.config import get_settings
from app.core.idempotency import create_idempotency_store
from app.core.logging import RequestIdMiddleware, configure_logging, get_logger
from app.core.rate_limit import create_rate_limiter
from app.db.base import engine
from app.llm.anthropic import make_llm_client

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: create and tear down shared resources."""
    configure_logging()
    settings = get_settings()
    validate_auth_settings(settings)  # refuse to start misconfigured
    # LLM_PROVIDER picks who writes answers and runs the answer check; embeddings
    # always come from local Ollama. Same factory as the evals, so a measured
    # setting is the served one. Fails at startup if the provider is misconfigured
    # (anthropic without ANTHROPIC_API_KEY) rather than on the first question.
    app.state.ollama = make_llm_client(settings)
    logger.info(
        "llm client ready",
        provider=settings.llm_provider,
        model=app.state.ollama.model_name,
        embedding_model=settings.embedding_model,
    )
    # None = limiting switched off; the dependency then lets everything by.
    app.state.rate_limiter = (
        create_rate_limiter(settings) if settings.rate_limit_enabled else None
    )
    # None = idempotency keys switched off; the header is then ignored.
    app.state.idempotency = (
        create_idempotency_store(settings) if settings.idempotency_enabled else None
    )
    yield
    if app.state.idempotency is not None:
        await app.state.idempotency.close()
    if app.state.rate_limiter is not None:
        await app.state.rate_limiter.close()
    await app.state.ollama.close()
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="AeroIntel API",
        description="Aviation maintenance RAG — grounded answers with citations.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(auth_router)
    app.include_router(v1_router)
    app.include_router(health_router)
    return app


app = create_app()
