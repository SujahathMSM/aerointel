"""Drive the whole eval harness in CI, with no models involved.

What this can and cannot prove. CI has Postgres but no Ollama, so it
cannot measure real retrieval quality — those numbers need EmbeddingGemma
and belong to a local `python -m evals.run_eval` run. What CI *can*
guarantee is that the harness itself still works: that the golden set
loads, expected chunks resolve to real rows, the pipeline is callable,
the metrics compute, and the pass/fail gate fires in the right direction.

That guarantee is the point. A silently broken eval harness is exactly how
a retrieval regression reaches main unnoticed — the scoreboard stops
working, nobody sees a red build, and quality drifts with no alarm.
"""

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.db.models import Base, DocumentChunk
from app.services.pipeline import retrieve
from app.services.reranking import NoReranker
from evals.metrics import aggregate, recall_at_k, reciprocal_rank
from evals.run_eval import (
    RETRIEVE_K,
    SCORED_K_VALUES,
    load_golden_set,
    resolve_expected_ids,
)

pytestmark = pytest.mark.integration

DIM = 768
MARKER = "eval_harness_test"


def unit_vector(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


class StubOllama:
    """Deterministic 'embeddings': each question maps to the axis of the
    chunk it should retrieve, so the harness has a knowable right answer
    without a real model."""

    def __init__(self, axis_for_text: dict[str, int]):
        self.axis_for_text = axis_for_text

    async def embed(self, texts):
        return [unit_vector(self.axis_for_text.get(t, 700)) for t in texts]


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
        sa.delete(DocumentChunk).where(
            DocumentChunk.metadata_["source"].astext == MARKER
        )
    )
    await session.commit()


def test_golden_set_loads_and_is_usable():
    items = load_golden_set()
    assert len(items) >= 10
    for item in items:
        assert item["question"].strip()
        assert item["expected"]["source"]
        assert item["expected"]["section"]


async def test_expected_ids_resolve_against_real_rows(db_session):
    chunk = DocumentChunk(
        content="Hydraulic filter contamination check.",
        metadata_={"source": MARKER, "section": "9.9"},
        embedding=unit_vector(0),
    )
    db_session.add(chunk)
    await db_session.commit()

    try:
        lookup = await resolve_expected_ids(db_session, load_golden_set())
        # The resolver keys on (source, section) — our marker row must be
        # findable exactly the way a golden item is looked up.
        assert lookup[(MARKER, "9.9")] == chunk.id
    finally:
        await _cleanup(db_session)


async def test_harness_scores_a_perfect_run(db_session):
    # Three chunks on orthogonal axes; each question's stub embedding points
    # straight at its own chunk, so a working harness must score 1.0.
    chunks = [
        DocumentChunk(
            content=f"Fixture chunk {i}.",
            metadata_={"source": MARKER, "section": str(i)},
            embedding=unit_vector(i),
        )
        for i in range(3)
    ]
    db_session.add_all(chunks)
    await db_session.commit()

    questions = {f"question about {i}": i for i in range(3)}
    ollama = StubOllama(questions)

    try:
        per_item = []
        for question, axis in questions.items():
            results = await retrieve(
                db_session,
                ollama,
                question,
                RETRIEVE_K,
                hybrid=False,
                reranker=NoReranker(),
            )
            ranked = [c.id for c, _ in results]
            expected_id = chunks[axis].id
            row = {"rr": reciprocal_rank(ranked, expected_id)}
            for k in SCORED_K_VALUES:
                row[f"recall@{k}"] = recall_at_k(ranked, expected_id, k)
            per_item.append(row)

        summary = aggregate(per_item, SCORED_K_VALUES)
        assert summary["recall@1"] == 1.0
        assert summary["mrr"] == 1.0
    finally:
        await _cleanup(db_session)


async def test_harness_detects_a_bad_run(db_session):
    # The gate must be able to FAIL, not just pass. Point every question at
    # an axis no chunk occupies: recall must collapse to 0.
    chunks = [
        DocumentChunk(
            content=f"Fixture chunk {i}.",
            metadata_={"source": MARKER, "section": str(i)},
            embedding=unit_vector(i),
        )
        for i in range(3)
    ]
    db_session.add_all(chunks)
    await db_session.commit()

    ollama = StubOllama({})  # every question -> unused axis 700

    try:
        results = await retrieve(
            db_session,
            ollama,
            "unrelated",
            RETRIEVE_K,
            hybrid=False,
            reranker=NoReranker(),
        )
        ranked = [c.id for c, _ in results]
        missing_id = -1  # an id that cannot be retrieved
        row = {"rr": reciprocal_rank(ranked, missing_id)}
        for k in SCORED_K_VALUES:
            row[f"recall@{k}"] = recall_at_k(ranked, missing_id, k)

        summary = aggregate([row], SCORED_K_VALUES)
        assert summary["recall@1"] == 0.0
        assert summary["mrr"] == 0.0
    finally:
        await _cleanup(db_session)
