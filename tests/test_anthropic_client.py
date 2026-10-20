"""The Anthropic provider: request shape, parsing, retries, and the key rule.

No network: httpx.MockTransport stands in for the API.
"""

import httpx
import pytest

from app.core.config import get_settings
from app.llm import anthropic as anthropic_module
from app.llm.anthropic import AnthropicChatClient, make_llm_client
from app.llm.ollama import OllamaClient


def _settings(**update):
    return get_settings().model_copy(
        update={"anthropic_api_key": "test-key", "ollama_max_retries": 3, **update}
    )


def _client(handler, **update):
    client = AnthropicChatClient(_settings(**update))
    client._api = httpx.AsyncClient(
        base_url="https://api.anthropic.test",
        transport=httpx.MockTransport(handler),
        headers=client._api.headers,
    )
    return client


async def test_chat_sends_system_and_user_and_reads_the_text():
    seen = {}

    def handler(request):
        seen["headers"] = request.headers
        seen["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "  Use .032 wire [1].  "}]},
        )

    client = _client(handler, anthropic_model="claude-haiku-5-5", temperature=0.2)
    answer = await client.chat("SYSTEM RULES", "CONTEXT ... QUESTION")
    await client.close()

    assert answer == "Use .032 wire [1]."
    assert seen["headers"]["x-api-key"] == "test-key"
    assert seen["headers"]["anthropic-version"] == anthropic_module.ANTHROPIC_VERSION
    body = seen["body"]
    assert '"model":"claude-haiku-5-5"' in body.replace(" ", "")
    assert '"system":"SYSTEMRULES"' in body.replace(" ", "")
    assert '"role":"user"' in body.replace(" ", "")
    # Live 2026-10-20: the API answers 400 "`temperature` is deprecated for
    # this model" when it is sent.
    assert "temperature" not in body


async def test_overloaded_is_retried_then_succeeds(monkeypatch):
    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("app.core.resilience.asyncio.sleep", no_sleep, raising=False)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(529, json={"error": {"type": "overloaded_error"}})
        return httpx.Response(200, json={"content": [{"type": "text", "text": "YES"}]})

    client = _client(handler)
    assert await client.chat("s", "u") == "YES"
    await client.close()
    assert len(calls) == 3


async def test_a_bad_request_fails_at_once_with_the_reason():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "model not found"}})

    client = _client(handler)
    with pytest.raises(RuntimeError, match="model not found"):
        await client.chat("s", "u")
    await client.close()
    assert len(calls) == 1


def test_no_key_no_client():
    settings = get_settings().model_copy(
        update={"llm_provider": "anthropic", "anthropic_api_key": None}
    )
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        make_llm_client(settings)


def test_the_default_provider_is_local():
    assert get_settings().llm_provider == "ollama"
    client = make_llm_client(get_settings())
    assert type(client) is OllamaClient
    assert client.model_name == get_settings().llm_model


async def test_a_cut_off_answer_is_marked_not_passed_off_as_complete():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "stop_reason": "max_tokens",
                "content": [{"type": "text", "text": "within conduits [3"}],
            },
        )

    client = _client(handler)
    answer = await client.chat("s", "u")
    await client.close()
    assert answer.startswith("within conduits [3")
    assert answer.endswith(anthropic_module.TRUNCATED_NOTE)


def test_the_output_limit_leaves_room_for_reasoning():
    assert get_settings().anthropic_max_tokens >= 4096
