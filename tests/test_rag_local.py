import re

import pytest
import sqlalchemy as sa
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.db.models import Base, DocumentChunk
from app.llm.ollama import OllamaClient
from app.services.generation import answer_question
from app.services.ingestion import ingest_chunk

pytestmark = pytest.mark.local

MARKER = "rag_local_test"

MATCHED_FIXTURE = (
    "If hydraulic pressure falls below 180 bar, inspect the hydraulic "
    "filter and check the engine-driven pump for cavitation."
)
MATCHED_QUESTION = "what should I check if hydraulic pressure drops below 180 bar?"
UNMATCHED_QUESTION = "what is the best recipe for chocolate sourdough bread?"

CITATION_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


@pytest.fixture
async def db_session():
    settings = get_settings()
    engine = create_async_engine(settings.database_url)

    async with engine.begin() as conn:
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
async def ollama():
    client = OllamaClient()
    yield client
    await client.close()


async def _cleanup(session):
    await session.execute(
        delete(DocumentChunk).where(DocumentChunk.metadata_["source"].astext == MARKER)
    )
    await session.commit()


async def test_grounded_answer_cites_context(db_session, ollama):
    await ingest_chunk(
        db_session,
        ollama,
        MATCHED_FIXTURE,
        {"source": MARKER},
    )

    try:
        result = await answer_question(db_session, ollama, MATCHED_QUESTION)

        assert not result.refused
        assert result.answer
        assert CITATION_RE.search(result.answer), (
            f"answer missing [n] citation: {result.answer}"
        )
    finally:
        await _cleanup(db_session)


async def test_unmatched_question_refuses_without_llm(db_session, ollama):
    await ingest_chunk(
        db_session,
        ollama,
        MATCHED_FIXTURE,
        {"source": MARKER},
    )

    try:
        result = await answer_question(db_session, ollama, UNMATCHED_QUESTION)

        assert result.refused
        assert result.answer is None
    finally:
        await _cleanup(db_session)
