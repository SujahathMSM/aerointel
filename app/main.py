from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.v1.router import router as v1_router
from app.core.logging import RequestIdMiddleware, configure_logging
from app.db.base import engine
from app.llm.ollama import OllamaClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: create and tear down shared resources."""
    configure_logging()
    app.state.ollama = OllamaClient()
    yield
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
