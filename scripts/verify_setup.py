"""Verify AeroIntel's local setup end-to-end.

Runs a sequence of checks against Ollama, PostgreSQL, pgvector, and the
document_chunks table. Each independent check runs regardless of earlier
failures, so a single run shows the whole picture. Checks with an unmet
prerequisite are skipped rather than reported as failures.

Closes the last open item on the README's Stage 1C checklist:
'PostgreSQL confirms the vector has 768 dimensions'.

Exits 0 if all checks pass, 1 otherwise. Cleans up its own test row.
"""

import os
import sys
import uuid
from typing import Callable

import httpx
import psycopg
from psycopg.types.json import Jsonb


OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "embeddinggemma"
EMBEDDING_DIMENSIONS = 768


results: list[tuple[str, str, str]] = []


def connection_string() -> str:
    for var in ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"):
        if var not in os.environ:
            raise RuntimeError(
                f"environment variable {var} is not set"
            )

    return (
        f"host=localhost "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ['POSTGRES_DB']} "
        f"user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


def get_embedding(text: str) -> list[float]:
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
        },
        timeout=30.0,
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


def vector_to_pgvector(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


# --- Checks -------------------------------------------------------------


def check_ollama_reachable() -> str:
    response = httpx.get(
        f"{OLLAMA_BASE_URL}/api/tags",
        timeout=5.0,
    )
    response.raise_for_status()
    return OLLAMA_BASE_URL


def check_model_available() -> str:
    response = httpx.get(
        f"{OLLAMA_BASE_URL}/api/tags",
        timeout=5.0,
    )
    response.raise_for_status()

    names = {m["name"] for m in response.json().get("models", [])}
    base_names = {n.split(":", 1)[0] for n in names}

    if EMBEDDING_MODEL not in names and EMBEDDING_MODEL not in base_names:
        raise RuntimeError(
            f"model '{EMBEDDING_MODEL}' not found. "
            f"Run: ollama pull {EMBEDDING_MODEL}"
        )

    return EMBEDDING_MODEL


def check_embedding_dimensions() -> str:
    embedding = get_embedding("verification probe")

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"expected {EMBEDDING_DIMENSIONS}, got {len(embedding)}"
        )

    return f"{len(embedding)} dimensions"


def check_postgres_reachable() -> str:
    with psycopg.connect(connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute("SHOW server_version")
            version = cur.fetchone()[0]

    return f"PostgreSQL {version}"


def check_vector_extension() -> str:
    with psycopg.connect(connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT extversion "
                "FROM pg_extension "
                "WHERE extname = 'vector'"
            )
            row = cur.fetchone()

    if not row:
        raise RuntimeError(
            "pgvector extension not enabled. "
            "Run: CREATE EXTENSION vector;"
        )

    return f"pgvector {row[0]}"


def check_document_chunks_schema() -> str:
    expected = {
        "id": "bigint",
        "content": "text",
        "metadata": "jsonb",
        "embedding": "USER-DEFINED",
        "created_at": "timestamp with time zone",
    }

    with psycopg.connect(connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'document_chunks'
                """
            )
            actual = dict(cur.fetchall())

    if not actual:
        raise RuntimeError(
            "table 'document_chunks' not found. "
            "Apply sql/001_create_document_chunks.sql."
        )

    missing = [c for c in expected if c not in actual]
    if missing:
        raise RuntimeError(f"missing columns: {missing}")

    mismatched = [
        f"{c} is {actual[c]}, expected {expected[c]}"
        for c in expected
        if actual[c] != expected[c]
    ]
    if mismatched:
        raise RuntimeError("; ".join(mismatched))

    return f"{len(expected)} columns match"


def check_cosine_operator() -> str:
    with psycopg.connect(connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT '[1,2,3]'::vector <=> '[1,2,3]'::vector")
            distance = cur.fetchone()[0]

    if distance != 0:
        raise RuntimeError(
            f"identical vectors returned distance {distance}"
        )

    return "identical vectors -> distance 0"


def check_roundtrip_vector_dims() -> str:
    """Insert a real embedding, then let PostgreSQL confirm its dimensions.

    This is the check the README's Stage 1C explicitly requires: not just
    that the client sends 768 values, but that PostgreSQL agrees after the
    round-trip. Uses a unique marker in metadata so cleanup is safe even
    if a previous run crashed.
    """
    embedding = get_embedding("stage 1C round-trip verification")
    marker = f"verify_setup_{uuid.uuid4().hex}"
    metadata = {"source": marker, "purpose": "verification"}

    with psycopg.connect(connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_chunks (content, metadata, embedding)
                VALUES (%s, %s, %s::vector)
                RETURNING id;
                """,
                (
                    "Round-trip verification chunk.",
                    Jsonb(metadata),
                    vector_to_pgvector(embedding),
                ),
            )
            inserted_id = cur.fetchone()[0]

            cur.execute(
                "SELECT vector_dims(embedding) "
                "FROM document_chunks "
                "WHERE id = %s",
                (inserted_id,),
            )
            dims = cur.fetchone()[0]

            cur.execute(
                "DELETE FROM document_chunks "
                "WHERE metadata ->> 'source' = %s",
                (marker,),
            )

    if dims != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"PostgreSQL sees {dims} dimensions, "
            f"expected {EMBEDDING_DIMENSIONS}"
        )

    return f"id={inserted_id}, vector_dims={dims}"


# --- Runner -------------------------------------------------------------


def run(
    label: str,
    fn: Callable[[], str],
    needs: list[str] | None = None,
) -> None:
    if needs:
        failed_needs = [
            n
            for n in needs
            if any(l == n and s != "PASS" for l, s, _ in results)
        ]
        if failed_needs:
            results.append(
                (label, "SKIP", f"needs {', '.join(failed_needs)}")
            )
            return

    try:
        detail = fn()
    except Exception as exc:
        results.append(
            (label, "FAIL", str(exc) or type(exc).__name__)
        )
        return

    results.append((label, "PASS", detail))


def print_results() -> None:
    width = max(len(label) for label, _, _ in results)

    for label, status, detail in results:
        line = f"  [{status}]  {label.ljust(width)}"
        if detail:
            line += f"  {detail}"
        print(line)

    print()

    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    skipped = sum(1 for _, s, _ in results if s == "SKIP")

    summary = f"{passed}/{len(results)} passed"
    if failed:
        summary += f", {failed} failed"
    if skipped:
        summary += f", {skipped} skipped"

    print(summary)


def main() -> None:
    print("Verifying AeroIntel setup...")
    print()

    run("Ollama reachable", check_ollama_reachable)
    run(
        "EmbeddingGemma available",
        check_model_available,
        needs=["Ollama reachable"],
    )
    run(
        "Embedding returns 768 dims",
        check_embedding_dimensions,
        needs=["EmbeddingGemma available"],
    )

    run("PostgreSQL reachable", check_postgres_reachable)
    run(
        "pgvector extension enabled",
        check_vector_extension,
        needs=["PostgreSQL reachable"],
    )
    run(
        "document_chunks schema matches",
        check_document_chunks_schema,
        needs=["PostgreSQL reachable"],
    )
    run(
        "cosine operator works",
        check_cosine_operator,
        needs=["pgvector extension enabled"],
    )

    run(
        "Round-trip vector_dims = 768",
        check_roundtrip_vector_dims,
        needs=[
            "Embedding returns 768 dims",
            "document_chunks schema matches",
            "pgvector extension enabled",
        ],
    )

    print_results()

    if any(status == "FAIL" for _, status, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()