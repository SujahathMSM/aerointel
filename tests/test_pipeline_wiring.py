"""Proves /v1/query and /v1/search actually call app.services.pipeline.retrieve
now, instead of the old direct app.services.retrieval.search_chunks — and
that wiring the pipeline in did not disturb the one property that matters
most: refusal stays structural (retrieve() returning nothing close enough
must still mean the LLM is never called).

No live Postgres/Ollama needed: retrieve() itself is replaced with a fake
via monkeypatch, so this tests wiring, not retrieval quality (that's
evals/run_eval.py's job, against a real database).
"""

from app.api.v1 import router as router_module
from app.core.config import get_settings
from app.schemas.search import SearchRequest
from app.services import generation


class _FakeChunk:
    def __init__(self, cid: int, content: str, source: str = "unit_test"):
        self.id = cid
        self.content = content
        self.metadata_ = {"source": source}


class _FakeOllama:
    """chat() is recorded so a test can assert it was, or wasn't, called."""

    def __init__(self, answer: str = "Answer with [1]."):
        self.answer = answer
        self.chat_calls: list[tuple[str, str]] = []

    async def chat(self, system: str, user: str) -> str:
        self.chat_calls.append((system, user))
        return self.answer


async def test_answer_question_goes_through_pipeline_retrieve(monkeypatch):
    captured = {}

    async def fake_retrieve(session, ollama, query, top_k=None, **kwargs):
        captured["args"] = (session, ollama, query, top_k)
        return [(_FakeChunk(1, "hydraulic pressure content"), 0.1)]

    monkeypatch.setattr(generation, "retrieve", fake_retrieve)

    ollama = _FakeOllama()
    result = await generation.answer_question(
        session="fake-session", ollama=ollama, question="what?", top_k=3
    )

    assert captured["args"] == ("fake-session", ollama, "what?", 3)
    assert result.refused is False
    assert result.answer == "Answer with [1]."
    assert ollama.chat_calls  # generation actually ran


async def test_answer_question_refusal_stays_structural(monkeypatch):
    # Distance above the ceiling -> refuse. The point of this test: after
    # rerouting through pipeline.retrieve(), the LLM must still never be
    # called on a refusal — that guarantee predates Phase 1 and must
    # survive it.
    async def fake_retrieve(session, ollama, query, top_k=None, **kwargs):
        return [(_FakeChunk(1, "irrelevant"), 0.99)]

    monkeypatch.setattr(generation, "retrieve", fake_retrieve)

    ollama = _FakeOllama()
    result = await generation.answer_question(
        session=None, ollama=ollama, question="what?"
    )

    assert result.refused is True
    assert ollama.chat_calls == []


async def test_search_endpoint_goes_through_pipeline_retrieve(monkeypatch):
    captured = {}

    async def fake_retrieve(session, ollama, query, top_k=None, **kwargs):
        captured["args"] = (query, top_k)
        return [(_FakeChunk(2, "some content"), 0.2)]

    monkeypatch.setattr(router_module, "retrieve", fake_retrieve)

    response = await router_module.search(
        SearchRequest(query="hydraulic pressure"),
        session="fake-session",
        ollama="fake-ollama",
    )

    # top_k=None on the request falls back to settings.top_k inside the
    # endpoint itself, so that's what retrieve() should have received.
    assert captured["args"] == ("hydraulic pressure", get_settings().top_k)
    assert response.query == "hydraulic pressure"
    assert len(response.results) == 1
    assert response.results[0].id == 2
    assert response.results[0].source == "unit_test"
