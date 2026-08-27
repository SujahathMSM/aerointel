"""Stage 1D - real pgvector semantic search from a user query.

Takes a query string from the command line, embeds it with EmbeddingGemma,
and asks PostgreSQL to rank the stored chunks by cosine distance.

    python scripts/search_embeddings.py "low hydraulic pressure"

If no query is supplied, falls back to a default so quick sanity runs
still work.
"""

import os
import sys

import httpx
import psycopg


OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "embeddinggemma"
EMBEDDING_DIMENSIONS = 768

DEFAULT_QUERY = "Aircraft engine shaking while ascending."
TOP_K = 5


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
    # sys.argv[0] is the script path, [1:] is the arg list. Join so a
    # multi-word query without shell quotes still works.
    argv_query = " ".join(sys.argv[1:]).strip()
    query = argv_query or DEFAULT_QUERY

    query_embedding = get_embedding(query)
    query_vector = vector_to_pgvector(query_embedding)

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
                SELECT
                    id,
                    content,
                    metadata,
                    embedding <=> %s::vector AS distance
                FROM document_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (
                    query_vector,
                    query_vector,
                    TOP_K,
                ),
            )

            results = cur.fetchall()

    print(f"Query: {query}")
    print()

    if not results:
        print("No chunks in the database yet.")
        return

    for rank, row in enumerate(results, start=1):
        chunk_id, content, metadata, distance = row

        print(f"{rank}. id={chunk_id}")
        print(f"   distance={distance:.4f}")
        print(f"   content={content}")
        print(f"   metadata={metadata}")
        print()


if __name__ == "__main__":
    main()