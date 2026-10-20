"""Answers and answer checks from an Anthropic model; embeddings stay local.

Only `chat()` goes to the API. Retrieval (EmbeddingGemma via Ollama, pgvector,
BM25) is unchanged, so a comparison with the local model changes exactly one
thing: the model that reads the passages. Selected with LLM_PROVIDER=anthropic.

This is a first slice of M2 for evaluation. Passages are sent to the API, so
use it only for documents that may leave the machine (the FAA manual is
public); the per-collection `local_only` flag arrives with M2 proper.
"""

import httpx

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.resilience import retry
from app.llm.ollama import OllamaClient

ANTHROPIC_VERSION = "2023-06-01"
TRUNCATED_NOTE = "[Answer cut off at the output token limit.]"

logger = get_logger(__name__)


class RetryableStatus(Exception):
    """429 / overloaded / 5xx: worth one more try after a back-off."""


_RETRY_STATUSES = {429, 500, 502, 503, 504, 529}
_CHAT_RETRYABLE = (httpx.ConnectError, httpx.ConnectTimeout, RetryableStatus)


class AnthropicChatClient(OllamaClient):
    """`chat()` from the Anthropic Messages API; `embed()` from local Ollama."""

    def __init__(self, settings: Settings | None = None):
        settings = settings or get_settings()
        if not settings.anthropic_api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic needs ANTHROPIC_API_KEY")
        super().__init__(settings)
        self._api = httpx.AsyncClient(
            base_url=settings.anthropic_base_url,
            timeout=httpx.Timeout(settings.ollama_timeout_seconds),
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )

    @property
    def model_name(self) -> str:
        return self.settings.anthropic_model

    async def chat(self, system: str, user: str) -> str:
        async def call() -> str:
            response = await self._api.post(
                "/v1/messages",
                json={
                    "model": self.settings.anthropic_model,
                    "max_tokens": self.settings.anthropic_max_tokens,
                    # No `temperature`: current Claude models reject it
                    # ("deprecated for this model"); they use their default.
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
            if response.status_code in _RETRY_STATUSES:
                raise RetryableStatus(f"Anthropic API {response.status_code}")
            if response.status_code != 200:
                # The error body names the problem (bad model id, no credit...);
                # it never contains the key.
                raise RuntimeError(
                    f"Anthropic chat failed ({response.status_code}): {response.text}"
                )
            data = response.json()
            blocks = data.get("content", [])
            text = "".join(
                block.get("text", "") for block in blocks if block.get("type") == "text"
            ).strip()
            if data.get("stop_reason") == "max_tokens":
                # Live 2026-10-20: 6 of 52 answers stopped mid-sentence at a
                # 1024-token limit. Never pass a cut answer off as complete.
                logger.warning(
                    "anthropic answer hit max_tokens",
                    model=self.settings.anthropic_model,
                    max_tokens=self.settings.anthropic_max_tokens,
                )
                text = f"{text}\n\n{TRUNCATED_NOTE}"
            return text

        return await self.breaker.call(
            lambda: retry(
                call,
                attempts=self.settings.ollama_max_retries,
                retry_on=_CHAT_RETRYABLE,
            )
        )

    async def chat_stream(self, system: str, user: str):  # type: ignore[override]
        raise NotImplementedError(
            "streaming from the Anthropic provider is not built yet (M2)"
        )
        yield ""  # pragma: no cover - makes this an async generator

    async def close(self) -> None:
        await self._api.aclose()
        await super().close()


def make_llm_client(settings: Settings | None = None) -> OllamaClient:
    """The client for the configured LLM_PROVIDER (embeddings are always local)."""
    settings = settings or get_settings()
    if settings.llm_provider == "anthropic":
        return AnthropicChatClient(settings)
    return OllamaClient(settings)
