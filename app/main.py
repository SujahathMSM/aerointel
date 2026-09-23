from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.v1.router import router as v1_router
from app.core.config import get_settings
from app.core.logging import RequestIdMiddleware, configure_logging
from app.core.rate_limit import create_rate_limiter
from app.db.base import engine
from app.llm.ollama import OllamaClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: create and tear down shared resources."""
    configure_logging()
    settings = get_settings()
    app.state.ollama = OllamaClient()
    # None = limiting switched off; the dependency then lets everything by.
    app.state.rate_limiter = (
        create_rate_limiter(settings) if settings.rate_limit_enabled else None
    )
    yield
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
    app.include_router(v1_router)
    app.include_router(health_router)
    return app


app = create_app()
