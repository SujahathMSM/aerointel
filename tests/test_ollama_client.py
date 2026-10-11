"""Retry policy: retry what retrying can fix, fail fast on what it can't."""

import httpx
import pytest

from app.core.config import get_settings
from app.llm.ollama import OllamaClient


def _client(handler) -> OllamaClient:
    settings = get_settings().model_copy(update={"ollama_max_retries": 2})
    client = OllamaClient(settings)
    client._client = httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    return client


async def test_embed_does_not_retry_a_404():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(404, json={"error": "model not found"})

    client = _client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await client.embed(["x"])
    assert calls == 1  # a missing model will not appear on retry


async def test_embed_retries_a_connection_failure():
    calls = 0
    dims = get_settings().embedding_dimensions

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json={"embeddings": [[0.0] * dims]})

    client = _client(handler)
    assert len(await client.embed(["x"])) == 1
    assert calls == 2


async def test_chat_does_not_retry_a_read_timeout():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow model", request=request)

    client = _client(handler)
    with pytest.raises(httpx.ReadTimeout):
        await client.chat("system", "user")
    assert calls == 1  # re-sending would stack a second generation
