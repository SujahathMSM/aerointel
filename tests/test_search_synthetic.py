import pytest
import sqlalchemy as sa
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.db.models import Base, DocumentChunk
from app.services.retrieval import search_chunks

pytestmark = pytest.mark.integration

DIM = 768
MARKER = "synthetic_search_test"


def unit_vector(i: int) -> list[float]:
    """Unit vector along axis i — orthogonal to the others."""
    v = [0.0] * DIM
    v[i] = 1.0
    return v


class FakeOllama:
    """Returns a fixed vector: lets us test pgvector without Ollama."""

    def __init__(self, vector: list[float]):
        self.vector = vector

    async def embed(self, texts):
        return [self.vector for _ in texts]


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


async def _cleanup(session):
    await session.execute(
        delete(DocumentChunk).where(DocumentChunk.metadata_["source"].astext == MARKER)
    )
    await session.commit()


async def test_pgvector_ranks_by_meaning(db_session):
    # three fixtures on clearly different "axes"
    fixtures = [
        DocumentChunk(
            content="hydraulic pressure fixture",
            metadata_={"source": MARKER, "fixture": 0},
            embedding=unit_vector(0),
        ),
        DocumentChunk(
            content="pressurization fixture",
            metadata_={"source": MARKER, "fixture": 1},
            embedding=unit_vector(1),
        ),
        DocumentChunk(
            content="brake temperature fixture",
            metadata_={"source": MARKER, "fixture": 2},
            embedding=unit_vector(2),
        ),
    ]
    db_session.add_all(fixtures)
    await db_session.commit()

    try:
        results = await search_chunks(
            db_session, FakeOllama(unit_vector(0)), "probe query", top_k=5
        )

        # fixture 0 must rank first with (near) zero distance
        assert results[0][0].metadata_["fixture"] == 0
        assert results[0][1] == pytest.approx(0.0, abs=1e-6)

        # results must be ordered by ascending distance
        distances = [distance for _, distance in results]
        assert distances == sorted(distances)
    finally:
        await _cleanup(db_session)


async def test_health_ready_with_db(db_session, client):
    from app.db.base import get_session
    from app.main import app

    async def override():
        yield db_session

    app.dependency_overrides[get_session] = override
    try:
        response = await client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}
    finally:
        app.dependency_overrides.clear()
