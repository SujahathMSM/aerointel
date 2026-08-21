import os

import httpx
import psycopg
from psycopg.types.json import Jsonb


OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "embeddinggemma"
EMBEDDING_DIMENSIONS = 768


def get_embedding(text: str) -> list[float]:
    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
        },
        timeout=30.0,
    )

    response.raise_for_status()

    data = response.json()
    embedding = data["embeddings"][0]

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSIONS} dimensions, "
            f"got {len(embedding)}"
        )

    return embedding


def vector_to_pgvector(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


def main() -> None:
    text = "Engine vibration detected during climb."

    metadata = {
        "source": "stage_1c_test",
        "category": "engine",
    }

    embedding = get_embedding(text)

    connection_string = (
        f"host=localhost "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ['POSTGRES_DB']} "
        f"user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )

    with psycopg.connect(connection_string) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_chunks (
                    content,
                    metadata,
                    embedding
                )
                VALUES (%s, %s, %s::vector)
                RETURNING id;
                """,
                (
                    text,
                    Jsonb(metadata),
                    vector_to_pgvector(embedding),
                ),
            )

            inserted_id = cur.fetchone()[0]

    print(f"Inserted chunk id: {inserted_id}")
    print(f"Embedding dimensions: {len(embedding)}")


if __name__ == "__main__":
    main()