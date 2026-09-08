import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.db.models import Base, DocumentChunk
from app.services.hybrid import hybrid_search, keyword_search, rrf_fuse

pytestmark = pytest.mark.integration

DIM = 768
MARKER = "hybrid_test"


def unit_vector(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


class FakeOllama:
    """Returns a fixed vector: tests pgvector fusion without Ollama."""

    def __init__(self, vector: list[float]):
        self.vector = vector

    async def embed(self, texts):
        return [self.vector for _ in texts]


# --- rrf_fuse (pure, no DB) ----------------------------------------------


def test_rrf_item_in_both_rankings_wins():
    # id 2 appears in both lists -> fused score 1/(k+1) + 1/(k+2),
    # higher than anything appearing once at rank 1.
    fused = rrf_fuse([[1, 2], [2, 3]], k=60)
    assert fused[0] == 2


def test_rrf_order_follows_rank_sum():
    # id 1: rank 1 in A only. id 2: rank 2 in A, rank 1 in B.
    # 1: 1/61 = 0.01639; 2: 1/62 + 1/61 = 0.03279 -> id 2 first.
    fused = rrf_fuse([[1, 2], [2, 3]], k=60)
    assert fused == [2, 1, 3]


def test_rrf_single_ranking_preserves_order():
    assert rrf_fuse([[5, 3, 8]], k=60) == [5, 3, 8]


def test_rrf_empty_rankings():
    assert rrf_fuse([[], []]) == []


# --- keyword + hybrid (needs Postgres with the 0002 migration) ------------


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
    from sqlalchemy import delete

    await session.execute(
        delete(DocumentChunk).where(DocumentChunk.metadata_["source"].astext == MARKER)
    )
    await session.commit()


async def test_keyword_search_finds_exact_terms(db_session):
    # Keyword search needs no embeddings at all — this chunk has none.
    chunk = DocumentChunk(
        content="Brake temperature exceeding 300 degrees requires a fuse "
        "plug inspection before the next departure.",
        metadata_={"source": MARKER},
        embedding=None,
    )
    db_session.add(chunk)
    await db_session.commit()

    def _ours(results):
        # The dev DB holds the real sample corpus too; scope to this
        # test's marker rows so assertions are isolated.
        return [(c, r) for c, r in results if c.metadata_.get("source") == MARKER]

    try:
        results = _ours(await keyword_search(db_session, "fuse plug", top_k=10))
        assert len(results) == 1
        assert results[0][0].id == chunk.id
        assert results[0][1] > 0  # ts_rank produced a score

        # stemming: "brakes" matches "Brake"
        results = _ours(await keyword_search(db_session, "brakes", top_k=10))
        assert any(c.id == chunk.id for c, _ in results)

        # no overlap with this chunk's words -> it is absent
        results = _ours(await keyword_search(db_session, "hydraulic pump", top_k=10))
        assert results == []
    finally:
        await _cleanup(db_session)


async def test_hybrid_search_fuses_and_backfills_distances(db_session):
    # Two chunks: one matches by meaning (synthetic vector), one by exact
    # keyword. Hybrid must surface both, with distances for each.
    semantic_chunk = DocumentChunk(
        content="Engine vibration detected during climb.",
        metadata_={"source": MARKER, "kind": "semantic"},
        embedding=unit_vector(0),
    )
    keyword_chunk = DocumentChunk(
        content="The fuse plug activation risk requires brake cooling.",
        metadata_={"source": MARKER, "kind": "keyword"},
        embedding=unit_vector(1),
    )
    db_session.add_all([semantic_chunk, keyword_chunk])
    await db_session.commit()

    try:
        # Query vector points at the semantic chunk's axis.
        results = await hybrid_search(
            db_session, FakeOllama(unit_vector(0)), "fuse plug", top_k=5
        )
        ids = [chunk.id for chunk, _ in results]
        assert semantic_chunk.id in ids
        assert keyword_chunk.id in ids

        # Every fused result carries a distance (backfilled for the
        # keyword-only hit), so the generation distance gate still works.
        for _chunk, distance in results:
            assert distance is not None
            assert 0.0 <= distance <= 2.0

        # The semantic match (distance ~0) must outrank the keyword-only
        # hit, which sits on an orthogonal axis (distance ~1).
        by_id = {chunk.id: d for chunk, d in results}
        assert by_id[semantic_chunk.id] < by_id[keyword_chunk.id]
    finally:
        await _cleanup(db_session)
