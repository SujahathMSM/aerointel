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

    # Context widening: when a retrieved chunk is cut from a PDF, also give the
    # model the next `context_neighbors` chunks of the same document (a hit often
    # stops mid-section; the answer continues in the chunk after it). 0 = off.
    # At most `context_max_chunks` chunks go into one prompt, hits included.
    context_neighbors: int = 0
    context_max_chunks: int = 8
    # Context window (tokens) the model is asked to use. None = Ollama's default.
    # Anything past the window is silently dropped by Ollama, so a prompt that
    # may not fit is logged. Raise this (e.g. 8192) before raising the two above.
    llm_num_ctx: int | None = None

    # Answerability: words in the question that appear in no stored chunk
    # ("Airbus", "Garmin") mean the documents cannot answer it, however close
    # the nearest chunk looks. "off" = skip the check; "flag" = report the
    # unknown words in the response; "refuse" = also refuse without calling the
    # model. Off by default, like the other retrieval switches. A word counts as
    # known when at least `answerability_min_chunks` chunks contain it. Measure
    # with `python -m evals.doc_eval` before choosing "refuse".
    answerability_mode: Literal["off", "flag", "refuse"] = "off"
    answerability_min_chunks: int = 1
    # "refuse" needs at least this many unknown words. 1 is strictest; 2 spares
    # a question that merely uses one word the manual does not (measure first).
    answerability_min_unknown: int = 1

    # Verification of the answer's numbers against the passages it cites.
    # "flag" = report numbers/units the cited passages do not contain;
    # "block" = also withhold the answer (non-streaming endpoint only; a stream
    # has already been sent, so it gets a `verification` event instead).
    verify_mode: Literal["off", "flag", "block"] = "flag"

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

    # Document ingestion (PDF upload -> text -> chunks -> embeddings).
    # Largest upload POST /v1/documents accepts, in megabytes.
    max_upload_mb: int = 300
    # Target chunk size in characters (about 300 tokens) and how much of the
    # previous chunk is repeated at the start of the next, so an answer that
    # straddles a boundary still lands whole in one chunk.
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 150
    # Also start a chunk at each numbered paragraph heading ("6-181. TITANIUM
    # ..."), so a chunk does not mix the end of one topic with the start of the
    # next, and fold sliver chunks into the one before. Off by default like the
    # other retrieval switches: it only affects documents ingested (or
    # re-ingested with `scripts/ingest_pdfs.py --replace`) while it is on.
    # Measure with `python -m evals.doc_eval` before and after.
    chunk_by_paragraph: bool = False
    # Chunks embedded per Ollama call.
    embed_batch_size: int = 8
    # OCR for pages with no text layer (scanned PDFs). Off by default: it needs
    # the tesseract program and PyMuPDF (see requirements-ocr.txt).
    ocr_enabled: bool = False
    ocr_language: str = "eng"

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
