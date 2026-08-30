from functools import lru_cache

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

    # Resilience
    ollama_timeout_seconds: float = 180.0
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
