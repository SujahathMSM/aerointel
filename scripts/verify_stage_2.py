"""Verify Stage 2 - grounded RAG generation refuses without context and cites with it.

Stage 1D proved pgvector ranks by meaning. Stage 2 layers on top:
  - a distance ceiling (MAX_DISTANCE) that turns "closest" into "relevant enough"
  - Qwen3, constrained by a system prompt to answer only from CONTEXT
  - a hard refusal path when no chunk clears the ceiling

This script asserts both halves of that contract:

  1. Insert one on-topic fixture under a unique marker.
  2. Matched question -> at least one chunk clears MAX_DISTANCE,
     Qwen3 produces an answer that carries a citation like [1].
  3. Unmatched question -> zero chunks clear MAX_DISTANCE,
     ask.py's refusal path fires, Qwen3 is never called.
  4. Clean up its own rows.

Retrieval is scoped to this run's marker so the assertions are not
affected by whatever else already sits in document_chunks.

Exits 0 if all checks pass, 1 otherwise.
"""

import os
import re
import sys
import uuid
from typing import Callable

import psycopg
from psycopg.types.json import Jsonb

from ask import (
    EMBEDDING_DIMENSIONS,
    LLM_MODEL,
    MAX_DISTANCE,
    build_prompt,
    generate,
    get_embedding,
    vector_to_pgvector,
)


MATCHED_FIXTURE = (
    "If hydraulic pressure falls below 180 bar, inspect the hydraulic "
    "filter and check the engine-driven pump for cavitation."
)

MATCHED_QUESTION = (
    "what should I check if hydraulic pressure drops below 180 bar?"
)
UNMATCHED_QUESTION = "what is the best recipe for chocolate sourdough bread?"

# Matches [1], [2], [1, 3], [2,3] - the citation shapes the system prompt
# tells Qwen3 to emit.
CITATION_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


results: list[tuple[str, str, str]] = []


def connection_string() -> str:
    for var in ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"):
        if var not in os.environ:
            raise RuntimeError(f"environment variable {var} is not set")
    return (
        f"host=localhost "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ['POSTGRES_DB']} "
        f"user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


# --- Retrieval scoped to this run's marker ------------------------------


def retrieve_scoped(
    question: str, marker: str
) -> list[tuple[int, str, dict, float]]:
    """Same shape as ask.retrieve_chunks, but restricted to this run's rows.

    Scoping isolates the unmatched-question assertion from whatever else
    already lives in document_chunks (real corpus, leftovers from a
    crashed run, other fixtures). If we searched the full table an
    unrelated real chunk could sneak in under MAX_DISTANCE and turn a
    genuine refusal into a false failure.
    """
    query_embedding = get_embedding(question)
    query_vector = vector_to_pgvector(query_embedding)

    with psycopg.connect(connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, metadata,
                       embedding <=> %s::vector AS distance
                FROM document_chunks
                WHERE embedding IS NOT NULL
                  AND metadata ->> 'source' = %s
                ORDER BY embedding <=> %s::vector
                LIMIT 5;
                """,
                (query_vector, marker, query_vector),
            )
            rows = cur.fetchall()

    return [
        (rid, content, metadata, float(distance))
        for rid, content, metadata, distance in rows
        if float(distance) <= MAX_DISTANCE
    ]


# --- Fixture ------------------------------------------------------------


def insert_fixture(marker: str) -> int:
    embedding = get_embedding(MATCHED_FIXTURE)
    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"expected {EMBEDDING_DIMENSIONS} dims, got {len(embedding)}"
        )

    metadata = {"source": marker}

    with psycopg.connect(connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_chunks (content, metadata, embedding)
                VALUES (%s, %s, %s::vector)
                RETURNING id;
                """,
                (
                    MATCHED_FIXTURE,
                    Jsonb(metadata),
                    vector_to_pgvector(embedding),
                ),
            )
            return cur.fetchone()[0]


def cleanup(marker: str) -> int:
    with psycopg.connect(connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM document_chunks "
                "WHERE metadata ->> 'source' = %s",
                (marker,),
            )
            return cur.rowcount


# --- Runner (identical shape to verify_stage_1d) ------------------------


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
            results.append((label, "SKIP", f"needs {', '.join(failed)}"))
            return

    try:
        detail = fn()
    except Exception as exc:
        results.append((label, "FAIL", str(exc) or type(exc).__name__))
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
    marker = f"verify_2_{uuid.uuid4().hex}"
    print(
        f"Verifying Stage 2 (marker={marker}, "
        f"MAX_DISTANCE={MAX_DISTANCE}, model={LLM_MODEL})..."
    )
    print()

    matched_chunks: list[tuple[int, str, dict, float]] = []
    matched_answer: str = ""

    def _insert() -> str:
        row_id = insert_fixture(marker)
        return f"fixture inserted (id={row_id})"

    def _matched_retrieval() -> str:
        nonlocal matched_chunks
        matched_chunks = retrieve_scoped(MATCHED_QUESTION, marker)
        if not matched_chunks:
            raise RuntimeError(
                f"matched question returned 0 chunks under "
                f"MAX_DISTANCE={MAX_DISTANCE} - fixture should have matched"
            )
        distances = ", ".join(f"{d:.4f}" for _, _, _, d in matched_chunks)
        return f"{len(matched_chunks)} chunk(s), distances: {distances}"

    def _matched_generation() -> str:
        nonlocal matched_answer
        prompt = build_prompt(MATCHED_QUESTION, matched_chunks)
        matched_answer = generate(prompt)
        if not matched_answer.strip():
            raise RuntimeError("Qwen3 returned an empty answer")
        return f"answer length {len(matched_answer)} chars"

    def _matched_citation() -> str:
        match = CITATION_RE.search(matched_answer)
        if not match:
            raise RuntimeError(
                "answer contains no [n] citation - grounding rule broken:\n"
                + matched_answer
            )
        return f"citation present: {match.group(0)}"

    def _unmatched_retrieval() -> str:
        chunks = retrieve_scoped(UNMATCHED_QUESTION, marker)
        if chunks:
            closest = min(d for _, _, _, d in chunks)
            raise RuntimeError(
                f"unmatched question returned {len(chunks)} chunk(s) "
                f"under MAX_DISTANCE={MAX_DISTANCE} "
                f"(closest={closest:.4f}) - refusal path would not fire"
            )
        return f"0 chunks under MAX_DISTANCE={MAX_DISTANCE} (refusal fires)"

    def _cleanup() -> str:
        removed = cleanup(marker)
        if removed != 1:
            raise RuntimeError(
                f"expected to delete 1 row, deleted {removed}"
            )
        return f"{removed} row(s) removed"

    try:
        run("Insert matched fixture", _insert)
        run(
            "Matched question retrieves at least one chunk",
            _matched_retrieval,
            needs=["Insert matched fixture"],
        )
        run(
            "Qwen3 produces an answer for the matched question",
            _matched_generation,
            needs=["Matched question retrieves at least one chunk"],
        )
        run(
            "Answer carries a [n] citation",
            _matched_citation,
            needs=["Qwen3 produces an answer for the matched question"],
        )
        run(
            "Unmatched question returns zero chunks (refusal path)",
            _unmatched_retrieval,
            needs=["Insert matched fixture"],
        )
    finally:
        # Cleanup is a check itself, but also a guarantee: even if a check
        # above raised something the runner did not catch, we still try to
        # delete our test row so the table stays clean.
        run("Cleanup removed the fixture", _cleanup)

    print_results()

    if any(status == "FAIL" for _, status, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()