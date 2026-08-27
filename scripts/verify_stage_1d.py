"""Verify Stage 1D - pgvector cosine search returns semantically correct order.

Stage 1C proved a single vector round-trips through PostgreSQL correctly.
Stage 1D is a stronger claim: given many vectors, the database's cosine
operator ranks them by meaning, not by luck. So this script:

  1. Inserts three fixture chunks about clearly different aviation topics.
  2. Runs a query whose meaning matches the first fixture.
  3. Asserts that fixture ranks #1.
  4. Cleans up its own rows.

Each run uses a unique metadata marker, so parallel or previously crashed
runs cannot poison each other.

Exits 0 if all checks pass, 1 otherwise.
"""

import os
import sys
import uuid
from typing import Callable

import httpx
import psycopg
from psycopg.types.json import Jsonb


OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "embeddinggemma"
EMBEDDING_DIMENSIONS = 768


# Three fixtures picked so their topics do not overlap. If pgvector search
# is working, a hydraulic-pressure query must rank fixture 0 first - not
# by keyword match (embeddings do not do keywords) but by meaning.
FIXTURES = [
    "If hydraulic pressure falls below 180 bar, inspect the hydraulic "
    "filter and check the engine-driven pump for cavitation.",
    "Cabin altitude above 10000 feet triggers automatic deployment of "
    "passenger oxygen masks.",
    "Brake temperature exceeding 300 degrees Celsius requires a cooling "
    "period before the next departure.",
]

PROBE_QUERY = "low hydraulic pressure - what to inspect?"
EXPECTED_TOP_INDEX = 0


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


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Batch embed several texts in one Ollama call.

    Sending one request instead of N cuts round-trip overhead and matches
    the batching pattern used in embedding_similarity.py.
    """
    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        },
        timeout=60.0,
    )
    response.raise_for_status()

    embeddings = response.json()["embeddings"]

    for embedding in embeddings:
        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"expected {EMBEDDING_DIMENSIONS} dimensions, "
                f"got {len(embedding)}"
            )

    return embeddings


def vector_to_pgvector(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


# --- Checks -------------------------------------------------------------


def insert_fixtures(marker: str) -> list[tuple[int, int]]:
    """Insert the three fixtures. Returns [(fixture_index, row_id), ...]."""
    all_embeddings = get_embeddings(FIXTURES)

    inserted: list[tuple[int, int]] = []

    with psycopg.connect(connection_string()) as conn:
        with conn.cursor() as cur:
            for fixture_index, (text, embedding) in enumerate(
                zip(FIXTURES, all_embeddings)
            ):
                metadata = {
                    "source": marker,
                    "fixture_index": fixture_index,
                }
                cur.execute(
                    """
                    INSERT INTO document_chunks (content, metadata, embedding)
                    VALUES (%s, %s, %s::vector)
                    RETURNING id;
                    """,
                    (
                        text,
                        Jsonb(metadata),
                        vector_to_pgvector(embedding),
                    ),
                )
                row_id = cur.fetchone()[0]
                inserted.append((fixture_index, row_id))

    return inserted


def run_search(marker: str) -> list[tuple[int, int, float]]:
    """Query pgvector, scoped to this run's fixtures.

    Scoping by metadata->>'source' means the test is not disturbed by any
    other rows already in document_chunks.

    Returns [(fixture_index, row_id, distance), ...] in the order
    PostgreSQL returned them.
    """
    query_embedding = get_embeddings([PROBE_QUERY])[0]
    query_vector = vector_to_pgvector(query_embedding)

    with psycopg.connect(connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    (metadata ->> 'fixture_index')::int AS fixture_index,
                    id,
                    embedding <=> %s::vector AS distance
                FROM document_chunks
                WHERE metadata ->> 'source' = %s
                ORDER BY embedding <=> %s::vector;
                """,
                (query_vector, marker, query_vector),
            )
            rows = cur.fetchall()

    return [(int(idx), rid, float(dist)) for idx, rid, dist in rows]


def cleanup(marker: str) -> int:
    with psycopg.connect(connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM document_chunks "
                "WHERE metadata ->> 'source' = %s",
                (marker,),
            )
            return cur.rowcount


# --- Runner -------------------------------------------------------------


def run(
    label: str,
    fn: Callable[[], str],
    needs: list[str] | None = None,
) -> None:
    if needs:
        failed = [
            n for n in needs
            if any(l == n and s != "PASS" for l, s, _ in results)
        ]
        if failed:
            results.append(
                (label, "SKIP", f"needs {', '.join(failed)}")
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
    marker = f"verify_1d_{uuid.uuid4().hex}"
    print(f"Verifying Stage 1D (marker={marker})...")
    print()

    # Defensive cleanup: if a previous run crashed after INSERT but before
    # DELETE, its rows would be sitting in the table under an old marker.
    # We only clean up rows tagged with our own marker, so this is safe
    # even if two runs happen at the same time.

    ordering: list[tuple[int, int, float]] = []

    def _insert() -> str:
        inserted = insert_fixtures(marker)
        return f"{len(inserted)} fixture(s) inserted"

    def _search() -> str:
        nonlocal ordering
        ordering = run_search(marker)
        return f"{len(ordering)} hit(s) returned"

    def _ordering() -> str:
        distances = [d for _, _, d in ordering]
        ok = all(a <= b for a, b in zip(distances, distances[1:]))
        if not ok:
            raise RuntimeError(
                "distances not monotonically increasing: "
                + ", ".join(f"{d:.4f}" for d in distances)
            )
        return "distances: " + ", ".join(f"{d:.4f}" for d in distances)

    def _top_hit() -> str:
        top_fixture, top_id, top_dist = ordering[0]
        if top_fixture != EXPECTED_TOP_INDEX:
            raise RuntimeError(
                f"top hit is fixture {top_fixture} (id={top_id}), "
                f"expected fixture {EXPECTED_TOP_INDEX}"
            )
        return f"fixture {top_fixture} (id={top_id}, distance={top_dist:.4f})"

    def _range() -> str:
        # Cosine distance is defined on [0, 2]. Anything outside means
        # bad math somewhere - normalization gone wrong, or a driver bug.
        bad = [(i, d) for i, _, d in ordering if not (0.0 <= d <= 2.0)]
        if bad:
            raise RuntimeError(f"out-of-range distances: {bad}")
        return "all distances in [0, 2]"

    def _cleanup() -> str:
        removed = cleanup(marker)
        expected = len(FIXTURES)
        if removed != expected:
            raise RuntimeError(
                f"expected to delete {expected} rows, deleted {removed}"
            )
        return f"{removed} row(s) removed"

    try:
        run("Insert 3 fixtures", _insert)
        run("pgvector search returns hits", _search,
            needs=["Insert 3 fixtures"])
        run("Results ordered by ascending distance", _ordering,
            needs=["pgvector search returns hits"])
        run("Top result is the semantically correct fixture", _top_hit,
            needs=["Results ordered by ascending distance"])
        run("Distances are valid cosine values", _range,
            needs=["pgvector search returns hits"])
    finally:
        # Cleanup is a check itself, but also a guarantee: even if a check
        # above raised something the runner did not catch, we still try to
        # delete our test rows so the table stays clean.
        run("Cleanup removed all fixtures", _cleanup)

    print_results()

    if any(status == "FAIL" for _, status, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()