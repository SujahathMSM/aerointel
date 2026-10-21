"""LLM_PROVIDER reaches the API, not just the evals.

Before this, app/main.py built OllamaClient() directly, so LLM_PROVIDER=anthropic
changed `live_eval` but every /v1/query was still answered by local Qwen. These
tests pin the startup wiring and that each answer names the model that wrote it.
"""

# Must come first: configures the test auth environment before `app` loads.
from tests.auth_helpers import TEST_JWT_SECRET  # noqa: F401  # isort: skip

import json

import pytest
from asgi_lifespan import LifespanManager

from app import main as main_module
from app.api.v1 import router as router_module
from app.core.config import get_settings
from app.llm.anthropic import AnthropicChatClient
from app.llm.ollama import OllamaClient
from app.main import app
from app.schemas.query import QueryRequest
from app.services import generation


def _use_settings(monkeypatch, **update):
    settings = get_settings().model_copy(update=update)
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    return settings


@pytest.mark.redis
async def test_the_api_starts_with_the_configured_provider(monkeypatch):
    settings = _use_settings(
        monkeypatch,
        llm_provider="anthropic",
        anthropic_api_key="test-key",
        anthropic_model="claude-test-model",
    )
    async with LifespanManager(app):
        client = app.state.ollama
        assert isinstance(client, AnthropicChatClient)
        assert client.model_name == "claude-test-model"
        # Embeddings stay local: same Ollama base URL and embedding model.
        assert str(client._client.base_url).rstrip("/") == settings.ollama_base_url
        assert client.settings.embedding_model == settings.embedding_model


@pytest.mark.redis
async def test_the_api_defaults_to_the_local_model(monkeypatch):
    _use_settings(monkeypatch, llm_provider="ollama")
    async with LifespanManager(app):
        assert type(app.state.ollama) is OllamaClient


async def test_anthropic_without_a_key_refuses_to_start(monkeypatch):
    _use_settings(monkeypatch, llm_provider="anthropic", anthropic_api_key="")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        async with LifespanManager(app):
            pass  # pragma: no cover - startup must fail first


class _Chunk:
    def __init__(self):
        self.id = 1
        self.content = "Safety wire .032 inch."
        self.metadata_ = {"source": "AC"}


class _Llm:
    model_name = "claude-test-model"


async def test_an_answer_names_the_model_that_wrote_it(monkeypatch):
    async def fake_answer(session, llm, question, top_k=None):
        return generation.AnswerResult(
            question=question,
            refused=False,
            answer="Use .032 inch wire [1].",
            chunks=[(_Chunk(), 0.2)],
            prompt_version="v1",
        )

    monkeypatch.setattr(router_module, "answer_question", fake_answer)
    response = await router_module.query(QueryRequest(question="q?"), None, _Llm())
    assert response.model == "claude-test-model"


async def test_a_refusal_names_no_model(monkeypatch):
    async def fake_answer(session, llm, question, top_k=None):
        return generation.AnswerResult(
            question=question, refused=True, refusal_reason="too_far"
        )

    monkeypatch.setattr(router_module, "answer_question", fake_answer)
    response = await router_module.query(QueryRequest(question="q?"), None, _Llm())
    assert response.model is None


async def _first_event(response) -> dict:
    async for raw in response.body_iterator:
        text = raw if isinstance(raw, str) else raw.decode()
        data = next(line for line in text.splitlines() if line.startswith("data:"))
        return json.loads(data[5:])
    raise AssertionError("no event")


@pytest.mark.parametrize(
    "refused, expected", [(False, "claude-test-model"), (True, None)]
)
async def test_the_stream_meta_names_the_model(monkeypatch, refused, expected):
    prepared = generation.PreparedAnswer(
        question="q?",
        refused=refused,
        chunks=[(_Chunk(), 0.2)],
        system=None if refused else "sys",
        prompt=None if refused else "prompt",
    )

    async def fake_prepare(session, llm, question, top_k=None):
        return prepared

    monkeypatch.setattr(router_module, "prepare_answer", fake_prepare)
    response = await router_module.query_stream(
        QueryRequest(question="q?"), None, _Llm()
    )
    meta = await _first_event(response)
    assert meta["model"] == expected
