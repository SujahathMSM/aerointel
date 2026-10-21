import json
from collections.abc import AsyncGenerator

import anyio
import httpx

from app.core.config import Settings, get_settings
from app.core.resilience import CircuitBreaker, retry

# Only retry failures that retrying can plausibly fix. A 404 (model not
# pulled) or a 400 will fail identically every time, so retrying just burns
# time. Embeds are fast, so a timeout there is worth retrying; a chat read
# timeout means the model is already slow, and re-sending the same prompt
# would stack a second generation on top of the first -- so for chat we only
# retry when the connection itself failed.
_EMBED_RETRYABLE = (httpx.TransportError,)
_CHAT_RETRYABLE = (httpx.ConnectError, httpx.ConnectTimeout)


class StreamTrimmer:
    """Trim a streamed answer the way `chat()` strips a whole one.

    Leading whitespace is dropped; trailing whitespace is held back until more
    text follows it, so the pieces joined together equal `text.strip()`. Shared
    by every provider's `chat_stream()`.
    """

    def __init__(self) -> None:
        self._started = False  # have we emitted any non-whitespace yet?
        self._pending = ""  # whitespace held back until we know more follows

    def feed(self, piece: str) -> str:
        """The part of `piece` that can be emitted now ("" = nothing yet)."""
        buffered = self._pending + piece
        body = buffered.rstrip()
        self._pending = buffered[len(body) :]
        if not self._started:
            body = body.lstrip()
            self._pending = self._pending if body else ""
        if body:
            self._started = True
        return body

    @property
    def started(self) -> bool:
        return self._started


class OllamaClient:
    """Async Ollama client: timeout + retry + circuit breaker built in."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = httpx.AsyncClient(
            base_url=self.settings.ollama_base_url,
            timeout=httpx.Timeout(self.settings.ollama_timeout_seconds),
        )
        self.breaker = CircuitBreaker(
            failure_threshold=self.settings.breaker_failure_threshold,
            recovery_seconds=self.settings.breaker_recovery_seconds,
        )

    @property
    def model_name(self) -> str:
        return self.settings.llm_model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async def call() -> list[list[float]]:
            response = await self._client.post(
                "/api/embed",
                json={"model": self.settings.embedding_model, "input": texts},
                timeout=self.settings.ollama_embed_timeout_seconds,
            )
            response.raise_for_status()
            return response.json()["embeddings"]

        embeddings = await self.breaker.call(
            lambda: retry(
                call,
                attempts=self.settings.ollama_max_retries,
                retry_on=_EMBED_RETRYABLE,
            )
        )

        for embedding in embeddings:
            if len(embedding) != self.settings.embedding_dimensions:
                raise ValueError(
                    f"expected {self.settings.embedding_dimensions} dims, "
                    f"got {len(embedding)}"
                )
        return embeddings

    def _options(self) -> dict:
        options: dict = {"temperature": self.settings.temperature}
        if self.settings.llm_num_ctx:
            options["num_ctx"] = self.settings.llm_num_ctx
        return options

    async def chat(self, system: str, user: str) -> str:
        async def call() -> str:
            response = await self._client.post(
                "/api/chat",
                json={
                    "model": self.settings.llm_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": self._options(),
                },
            )
            if response.status_code != 200:
                # Ollama's error body is more useful than the bare status
                raise RuntimeError(
                    f"Ollama chat failed ({response.status_code}): {response.text}"
                )
            return response.json()["message"]["content"].strip()

        return await self.breaker.call(
            lambda: retry(
                call,
                attempts=self.settings.ollama_max_retries,
                retry_on=_CHAT_RETRYABLE,
            )
        )

    async def chat_stream(self, system: str, user: str) -> AsyncGenerator[str, None]:
        """Yield the answer as it is generated, a few tokens at a time.

        Same prompt, model and options as `chat()`, and the pieces joined
        together equal what `chat()` would return (leading/trailing whitespace
        is trimmed the same way).

        Resilience applies to *opening* the stream -- timeout, retry on
        connection failure, circuit breaker -- exactly as for `chat()`. Once
        tokens are flowing, a retry would restart the answer from the top and
        show the caller duplicated text, so a failure mid-stream is raised
        as-is. The per-read timeout (`ollama_timeout_seconds`) then acts as an
        "Ollama went quiet" detector between chunks.
        """

        async def open_stream() -> httpx.Response:
            request = self._client.build_request(
                "POST",
                "/api/chat",
                json={
                    "model": self.settings.llm_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": True,
                    "options": self._options(),
                },
            )
            response = await self._client.send(request, stream=True)
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")
                await response.aclose()
                raise RuntimeError(
                    f"Ollama chat failed ({response.status_code}): {body}"
                )
            return response

        response = await self.breaker.call(
            lambda: retry(
                open_stream,
                attempts=self.settings.ollama_max_retries,
                retry_on=_CHAT_RETRYABLE,
            )
        )

        trimmer = StreamTrimmer()
        try:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(
                        f"Ollama chat failed mid-stream: {data['error']}"
                    )
                body = trimmer.feed(data.get("message", {}).get("content", ""))
                if body:
                    yield body
                if data.get("done"):
                    break
        finally:
            # If the caller went away we may be running inside a cancelled
            # scope; shield the close so the connection is really released.
            with anyio.CancelScope(shield=True):
                await response.aclose()

    async def close(self) -> None:
        await self._client.aclose()
