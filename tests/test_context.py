"""Context widening: the chunk(s) after each hit join the prompt.

- plan_neighbours is pure: which following chunks, in what order, within what budget.
- expand_with_neighbours is checked against a real Postgres (marked `integration`).
- prepare_answer is checked with a fake session: neighbours come after the
  distance gate (so they can never turn a refusal into an answer) and before
  the session is released.
- the Ollama client sends num_ctx only when it is configured.
"""

import pytest
import sqlalchemy as sa
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.db.models import Base, Document, DocumentChunk
from app.llm.ollama import OllamaClient
from app.services import context, generation
from app.services.context import expand_with_neighbours, plan_neighbours


class Chunk:
    def __init__(self, cid, document_id=1, index=0, content="text", page=1):
        self.id = cid
        self.document_id = document_id
        self.chunk_index = index
        self.content = content
        self.page = page
        self.page_label = None
        self.metadata_ = {"source": "manual"}


# ------------------------------------------------------------------ the plan


def test_each_hit_gets_the_chunk_after_it():
    hits = [(Chunk(1, index=10), 0.1), (Chunk(2, document_id=2, index=4), 0.2)]
    assert plan_neighbours(hits, neighbours=1, max_chunks=8) == {
        1: [(1, 11)],
        2: [(2, 5)],
    }


def test_two_neighbours_go_in_reading_order():
    plan = plan_neighbours([(Chunk(1, index=10), 0.1)], neighbours=2, max_chunks=8)
    assert plan == {1: [(1, 11), (1, 12)]}


def test_a_neighbour_that_is_already_a_hit_is_not_added_twice():
    hits = [(Chunk(1, index=10), 0.1), (Chunk(2, index=11), 0.2)]
    plan = plan_neighbours(hits, neighbours=1, max_chunks=8)
    assert plan == {2: [(1, 12)]}  # 11 is already in the results


def test_the_budget_counts_the_hits_and_limits_the_neighbours():
    hits = [(Chunk(i, index=i * 10), 0.1) for i in range(1, 6)]
    plan = plan_neighbours(hits, neighbours=1, max_chunks=7)
    assert sum(len(keys) for keys in plan.values()) == 2  # 7 - 5 hits


def test_the_budget_is_spread_over_hits_not_spent_on_the_best_one():
    hits = [(Chunk(1, index=10), 0.1), (Chunk(2, index=20), 0.2)]
    plan = plan_neighbours(hits, neighbours=2, max_chunks=4)
    # two neighbours to spend: one each, not both for hit 1
    assert plan == {1: [(1, 11)], 2: [(1, 21)]}


def test_no_budget_left_means_no_neighbours():
    hits = [(Chunk(i, index=i), 0.1) for i in range(1, 9)]
    assert plan_neighbours(hits, neighbours=1, max_chunks=8) == {}


def test_plain_text_chunks_have_no_position_and_are_left_alone():
    plain = Chunk(1, document_id=None, index=None)
    assert plan_neighbours([(plain, 0.1)], neighbours=1, max_chunks=8) == {}


def test_neighbours_are_capped():
    plan = plan_neighbours([(Chunk(1, index=0), 0.1)], neighbours=99, max_chunks=999)
    assert len(plan[1]) == context.MAX_NEIGHBOURS


def test_zero_neighbours_plans_nothing():
    assert plan_neighbours([(Chunk(1), 0.1)], neighbours=0, max_chunks=8) == {}


# ------------------------------------------------------- prepare_answer wiring


class FakeSession:
    def __init__(self):
        self.events: list[str] = []

    async def commit(self):
        self.events.append("commit")


@pytest.fixture
def settings(monkeypatch):
    base = get_settings()
    values = base.model_copy(update={"context_neighbors": 1, "context_max_chunks": 8})
    monkeypatch.setattr(generation, "get_settings", lambda: values)
    return values


async def test_neighbours_join_the_prompt_after_the_gate_and_before_the_commit(
    settings, monkeypatch
):
    session = FakeSession()
    hit = Chunk(1, index=5, content="step c ends here", page=7)
    follower = Chunk(2, index=6, content="step d continues", page=8)

    async def fake_retrieve(session, ollama, question, top_k=None, **kw):
        return [(hit, 0.2), (Chunk(9, index=99, content="far away"), 0.9)]

    async def fake_expand(sess, results, neighbours, max_chunks):
        session.events.append("expand")
        assert [c.id for c, _ in results] == [1]  # the 0.9 chunk never got through
        return [*results, (follower, 0.2)]

    monkeypatch.setattr(generation, "retrieve", fake_retrieve)
    monkeypatch.setattr(generation, "expand_with_neighbours", fake_expand)

    prepared = await generation.prepare_answer(session, object(), "question?")

    assert prepared.refused is False
    assert [c.id for c, _ in prepared.chunks] == [1, 2]
    assert "step c ends here" in prepared.prompt
    assert "step d continues" in prepared.prompt
    assert session.events == ["expand", "commit"]


async def test_widening_cannot_turn_a_refusal_into_an_answer(settings, monkeypatch):
    async def fake_retrieve(session, ollama, question, top_k=None, **kw):
        return [(Chunk(1, content="unrelated"), 0.95)]

    async def never(*args, **kwargs):
        raise AssertionError("expansion ran for a refused question")

    monkeypatch.setattr(generation, "retrieve", fake_retrieve)
    monkeypatch.setattr(generation, "expand_with_neighbours", never)

    prepared = await generation.prepare_answer(FakeSession(), object(), "question?")
    assert prepared.refused is True
    assert prepared.prompt is None


async def test_with_widening_off_nothing_is_fetched(monkeypatch):
    off = get_settings().model_copy(update={"context_neighbors": 0})
    monkeypatch.setattr(generation, "get_settings", lambda: off)

    async def fake_retrieve(session, ollama, question, top_k=None, **kw):
        return [(Chunk(1, content="a hit"), 0.1)]

    async def never(*args, **kwargs):
        raise AssertionError("expansion ran although it is switched off")

    monkeypatch.setattr(generation, "retrieve", fake_retrieve)
    monkeypatch.setattr(generation, "expand_with_neighbours", never)

    prepared = await generation.prepare_answer(FakeSession(), object(), "q?")
    assert [c.id for c, _ in prepared.chunks] == [1]


# ------------------------------------------------------ context window guard


def _with_window(monkeypatch, tokens):
    values = get_settings().model_copy(update={"llm_num_ctx": tokens})
    monkeypatch.setattr(generation, "get_settings", lambda: values)


def test_a_prompt_that_fits_is_not_flagged(monkeypatch):
    _with_window(monkeypatch, 4096)
    assert generation.warn_if_prompt_may_not_fit("x" * 500, "y" * 5000) is False


def test_a_prompt_near_the_window_is_flagged(monkeypatch):
    _with_window(monkeypatch, 4096)
    # 4096 tokens ~ 14,300 characters at 3.5 chars/token; 85% is ~12,100
    assert generation.warn_if_prompt_may_not_fit("x" * 500, "y" * 13000) is True


def test_a_bigger_window_makes_the_same_prompt_fit(monkeypatch):
    _with_window(monkeypatch, 8192)
    assert generation.warn_if_prompt_may_not_fit("x" * 500, "y" * 13000) is False


def test_unset_window_is_judged_against_the_ollama_default(monkeypatch):
    _with_window(monkeypatch, None)
    assert generation.warn_if_prompt_may_not_fit("x", "y" * 13000) is True


# ------------------------------------------------------------- ollama options


def _client(**overrides):
    settings = get_settings().model_copy(update=overrides)
    client = OllamaClient.__new__(OllamaClient)
    client.settings = settings
    return client


def test_num_ctx_is_sent_only_when_configured():
    assert "num_ctx" not in _client(llm_num_ctx=None)._options()
    assert _client(llm_num_ctx=8192)._options()["num_ctx"] == 8192
    assert (
        _client(llm_num_ctx=8192)._options()["temperature"]
        == get_settings().temperature
    )


def test_the_new_settings_default_to_the_old_behaviour():
    fresh = Settings(postgres_password="x", _env_file=None)
    assert fresh.context_neighbors == 0
    assert fresh.llm_num_ctx is None


# ------------------------------------------------- integration: real Postgres

integration = pytest.mark.integration
TITLE = "context_test_manual"


@pytest.fixture
async def db_session():
    engine = create_async_engine(get_settings().database_url)
    async with engine.begin() as conn:
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        await session.execute(delete(Document).where(Document.title == TITLE))
        await session.commit()
        yield session
        await session.rollback()
        await session.execute(delete(Document).where(Document.title == TITLE))
        await session.commit()
    await engine.dispose()


async def _manual(session, n_chunks: int):
    doc = Document(
        filename="m.pdf",
        title=TITLE,
        content_hash="context-test-hash",
        byte_size=1,
        page_count=n_chunks,
        chunk_count=n_chunks,
        ocr_pages=0,
    )
    session.add(doc)
    await session.flush()
    chunks = [
        DocumentChunk(
            content=f"chunk number {i}",
            metadata_={"source": TITLE},
            document_id=doc.id,
            chunk_index=i,
            page=i + 1,
        )
        for i in range(n_chunks)
    ]
    session.add_all(chunks)
    await session.commit()
    return chunks


@integration
async def test_real_database_neighbour_follows_its_hit(db_session):
    chunks = await _manual(db_session, 6)

    expanded = await expand_with_neighbours(
        db_session, [(chunks[2], 0.2), (chunks[5], 0.3)], neighbours=1, max_chunks=8
    )

    # hit 2 -> 3 follows it; chunk 5 is the last one, so it has no follower
    assert [c.chunk_index for c, _ in expanded] == [2, 3, 5]
    assert [d for _, d in expanded] == [
        0.2,
        0.2,
        0.3,
    ]  # a neighbour inherits its hit's distance


@integration
async def test_real_database_two_neighbours_in_reading_order(db_session):
    chunks = await _manual(db_session, 6)

    expanded = await expand_with_neighbours(
        db_session, [(chunks[1], 0.1)], neighbours=2, max_chunks=8
    )

    assert [c.chunk_index for c, _ in expanded] == [1, 2, 3]


@integration
async def test_real_database_nothing_to_add_returns_the_hits_unchanged(db_session):
    chunks = await _manual(db_session, 3)
    hits = [(chunks[0], 0.1)]

    assert await expand_with_neighbours(db_session, hits, 0, 8) == hits
    assert await expand_with_neighbours(db_session, [], 1, 8) == []


@integration
async def test_real_database_other_documents_are_never_mixed_in(db_session):
    chunks = await _manual(db_session, 4)
    other = Document(
        filename="o.pdf",
        title=TITLE,
        content_hash="context-test-hash-2",
        byte_size=1,
        page_count=1,
        chunk_count=1,
        ocr_pages=0,
    )
    db_session.add(other)
    await db_session.flush()
    db_session.add(
        DocumentChunk(
            content="other manual, same index",
            metadata_={"source": TITLE},
            document_id=other.id,
            chunk_index=1,
            page=1,
        )
    )
    await db_session.commit()

    expanded = await expand_with_neighbours(
        db_session, [(chunks[0], 0.1)], neighbours=1, max_chunks=8
    )

    assert [c.content for c, _ in expanded] == ["chunk number 0", "chunk number 1"]
