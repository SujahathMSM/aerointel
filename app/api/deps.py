from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_session
from app.llm.ollama import OllamaClient


def get_ollama(request: Request) -> OllamaClient:
    """The shared OllamaClient lives on app.state (created at startup)."""
    return request.app.state.ollama


SessionDep = Annotated[AsyncSession, Depends(get_session)]
OllamaDep = Annotated[OllamaClient, Depends(get_ollama)]
