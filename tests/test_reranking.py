import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.db.models import Base, DocumentChunk
from app.services.pipeline import retrieve
from app.services.reranking import (
    CrossEncoderReranker,
    NoReranker,
    build_reranker,
    rerank,
)

DIM = 768
MARKER = "rerank_test"


def unit_vector(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


class FakeOllama:
    def __init__(self, vector: list[float]):
        self.vector = vector

    async def embed(self, texts):
        return [self.vector for _ in texts]


class KeywordReranker:
    """Fake cross-encoder: scores by exact substring presence."""

    def __init__(self, keyword: str):
        self.keyword = keyword

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [10.0 if self.keyword in d else 0.0 for d in documents]


# --- rerank() (pure) ------------------------------------------------------


class _FakeChunk:
    def __init__(self, cid: int, content: str):
        self.id = cid
        self.content = content


def _results(*pairs: tuple[int, str, float]) -> list[tuple]:
    return [(_FakeChunk(cid, content), dist) for cid, content, dist in pairs]


def test_rerank_reorders_by_score():
    results = _results((1, "alpha", 0.1), (2, "beta", 0.2), (3, "gamma", 0.3))
    reranker = KeywordReranker("gamma")
    out = rerank("q", results, reranker, top_k=3)
    assert [chunk.id for chunk, _ in out] == [3, 1, 2]


def test_rerank_truncates_to_top_k():
    results = _results((1, "a", 0.1), (2, "b", 0.2), (3, "c", 0.3))
    out = rerank("q", results, KeywordReranker("c"), top_k=2)
    assert len(out) == 2
    assert out[0][0].id == 3


def test_rerank_carries_distances_untouched():
    results = _results((1, "alpha", 0.123), (2, "beta", 0.456))
    out = rerank("q", results, KeywordReranker("beta"), top_k=2)
    by_id = {chunk.id: dist for chunk, dist in out}
    assert by_id[1] == 0.123
    assert by_id[2] == 0.456


def test_rerank_empty_results():
    assert rerank("q", [], KeywordReranker("x"), top_k=5) == []


# --- factory ---------------------------------------------------------------


def test_build_reranker_none():
    assert isinstance(build_reranker("none", "any-model"), NoReranker)


def test_build_reranker_crossencoder_is_lazy():
    # Constructing must not import torch or load the model.
    r = build_reranker("crossencoder", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert isinstance(r, CrossEncoderReranker)
    assert r._model is None


def test_build_reranker_unknown_fails_loudly():
    with pytest.raises(ValueError, match="unknown reranker"):
        build_reranker("magic", "any-model")


def test_no_reranker_scores_descending():
    assert NoReranker().score("q", ["a", "b", "c"]) == [3, 2, 1]


# --- pipeline (needs Postgres) ---------------------------------------------


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


@pytest.mark.integration
async def test_pipeline_reranker_promotes_keyword_match(db_session):
    # Three chunks on orthogonal axes. Vector search (query on axis 0)
    # puts chunk A first. The fake cross-encoder prefers the chunk whose
    # text contains "fuse plug" — the pipeline must promote it to rank 1.
    chunk_a = DocumentChunk(
        content="Engine vibration detected during climb.",
        metadata_={"source": MARKER},
        embedding=unit_vector(0),
    )
    chunk_b = DocumentChunk(
        content="Brake cooling protects the fuse plug from activation.",
        metadata_={"source": MARKER},
        embedding=unit_vector(1),
    )
    chunk_c = DocumentChunk(
        content="Cabin altitude above 10000 feet deploys oxygen masks.",
        metadata_={"source": MARKER},
        embedding=unit_vector(2),
    )
    db_session.add_all([chunk_a, chunk_b, chunk_c])
    await db_session.commit()

    ours = {chunk_a.id, chunk_b.id, chunk_c.id}

    def _ours(results):
        # A dev database also holds the real sample corpus, which would
        # otherwise crowd these fixtures out of a small top_k. Scope the
        # assertions to this test's own rows (same pattern as test_hybrid).
        return [(c, d) for c, d in results if c.id in ours]

    # Fetch wider than the 3 fixtures so they are reachable even when the
    # table already holds other rows.
    probe_k = 50

    try:
        # Baseline: no reranker -> pure vector order (A first).
        baseline = _ours(
            await retrieve(
                db_session,
                FakeOllama(unit_vector(0)),
                "probe",
                top_k=probe_k,
                hybrid=False,
                reranker=NoReranker(),
            )
        )
        assert len(baseline) == 3
        assert baseline[0][0].id == chunk_a.id

        # With the fake cross-encoder: B promoted ahead of the other two.
        reranked = _ours(
            await retrieve(
                db_session,
                FakeOllama(unit_vector(0)),
                "probe",
                top_k=probe_k,
                hybrid=False,
                reranker=KeywordReranker("fuse plug"),
            )
        )
        assert len(reranked) == 3
        assert reranked[0][0].id == chunk_b.id

        # Distances still travel with every result (distance gate intact).
        for _chunk, distance in reranked:
            assert distance is not None
    finally:
        await _cleanup(db_session)


@pytest.mark.local
async def test_crossencoder_scores_real_pairs():
    pytest.importorskip("sentence_transformers")
    reranker = CrossEncoderReranker(get_settings().reranker_model)
    scores = reranker.score(
        "what to check when hydraulic pressure drops?",
        [
            "If hydraulic pressure falls below 180 bar, inspect the filter.",
            "Passenger oxygen masks deploy above 10000 feet cabin altitude.",
        ],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]  # the on-topic doc must score higher
