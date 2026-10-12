from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All app config, loaded from environment variables and .env."""

    # Postgres (password is a secret: no default, app refuses to start without it)
    postgres_db: str = "aerointel"
    postgres_user: str = "aerointel"
    postgres_password: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "embeddinggemma"
    llm_model: str = "qwen3:4b-instruct"
    embedding_dimensions: int = 768

    # Retrieval / generation
    top_k: int = 5
    max_distance: float = 0.48
    temperature: float = 0.2

    # Hybrid retrieval (Phase 1): BM25-style keyword search fused with
    # vector search via Reciprocal Rank Fusion. Off by default so the
    # vector-only baseline stays the default behavior until evals prove
    # hybrid is better.
    hybrid_enabled: bool = False
    rrf_k: int = 60

    # Reranking (Phase 1): rescore the top-N candidates with a
    # cross-encoder that reads the full text. "none" keeps the baseline.
    # The cross-encoder needs requirements-rerank.txt installed; it is
    # lazy-imported so the app runs fine without it while disabled.
    reranker: str = "none"  # "none" | "crossencoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20

    # Prompt registry (Phase 1): prompts live as versioned files under
    # app/prompts/. None = use the newest version on disk; pin a version
    # ("v1") to freeze behavior while a newer one is being evaluated.
    prompt_version: str | None = None

    # Rate limiting (Phase 2): token bucket per client, stored in Redis so
    # every worker/replica shares one budget. Defaults: bursts of up to 20
    # requests, then a sustained 30/minute (0.5 tokens/second).
    redis_url: str = "redis://localhost:6379/0"
    rate_limit_enabled: bool = True
    rate_limit_capacity: int = 20
    rate_limit_refill_per_second: float = 0.5
    rate_limit_key_prefix: str = "ratelimit"
    # Redis down: True = let requests through (limiter is protection, not a
    # dependency), False = 503 everything until Redis is back.
    rate_limit_fail_open: bool = True

    # Authentication (Phase 2): self-issued JWTs. The secret has no default --
    # the API refuses to start without one while auth is on (checked at API
    # startup, not here, so alembic and scripts don't need it).
    auth_enabled: bool = True
    jwt_secret: str | None = None
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    jwt_issuer: str = "aerointel"
    jwt_audience: str = "aerointel-api"
    jwt_ttl_seconds: int = 3600
    # JSON: {"client-id": {"secret": "...", "scopes": ["query", "ingest"]}}
    auth_clients: str = ""

    # Idempotency keys (Phase 2): an `Idempotency-Key` header on POST /v1/ingest
    # makes retries safe. Finished responses are remembered for the TTL; a
    # request still running is guarded by the (shorter) lock, which must
    # outlast the slowest legitimate ingest (embed timeout x retries).
    idempotency_enabled: bool = True
    idempotency_ttl_seconds: int = 86400
    idempotency_lock_seconds: int = 150
    idempotency_key_prefix: str = "idempotency"

    # Resilience
    ollama_timeout_seconds: float = 180.0
    # Embedding one short string should take well under a second; waiting
    # the full chat timeout on a hung embed call only ties up the request.
    ollama_embed_timeout_seconds: float = 30.0
    ollama_max_retries: int = 3
    breaker_failure_threshold: int = 5
    breaker_recovery_seconds: float = 30.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def database_url(self) -> str:
        """Async URL used by the app (asyncpg driver)."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Sync URL used by Alembic migrations (psycopg driver)."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Cached so every import shares one Settings instance."""
    # password comes from env at runtime, not the constructor
    return Settings()  # type: ignore[call-arg]
