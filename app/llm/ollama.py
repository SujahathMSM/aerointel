import httpx

from app.core.config import Settings, get_settings
from app.core.resilience import CircuitBreaker, retry


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

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async def call() -> list[list[float]]:
            response = await self._client.post(
                "/api/embed",
                json={"model": self.settings.embedding_model, "input": texts},
            )
            response.raise_for_status()
            return response.json()["embeddings"]

        embeddings = await self.breaker.call(
            lambda: retry(call, attempts=self.settings.ollama_max_retries)
        )

        for embedding in embeddings:
            if len(embedding) != self.settings.embedding_dimensions:
                raise ValueError(
                    f"expected {self.settings.embedding_dimensions} dims, "
                    f"got {len(embedding)}"
                )
        return embeddings

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
                    "options": {"temperature": self.settings.temperature},
                },
            )
            if response.status_code != 200:
                # Ollama's error body is more useful than the bare status
                raise RuntimeError(
                    f"Ollama chat failed ({response.status_code}): {response.text}"
                )
            return response.json()["message"]["content"].strip()

        return await self.breaker.call(
            lambda: retry(call, attempts=self.settings.ollama_max_retries)
        )

    async def close(self) -> None:
        await self._client.aclose()
