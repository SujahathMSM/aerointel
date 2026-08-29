"""Load a small set of real aviation maintenance chunks.

Right now document_chunks has one usable row. Testing search or ask.py
against one row can't tell you anything about ranking - there's nothing
to rank against. This script inserts several chunks on different topics,
so a query has to actually pick the right one out of several candidates.

Synthetic content, written for this project. Not real aircraft manual
text and not airworthiness information.

    python scripts/load_sample_data.py
"""

import os

import httpx
import psycopg
from psycopg.types.json import Jsonb


OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "embeddinggemma"
EMBEDDING_DIMENSIONS = 768


# Each entry: (content, metadata). Metadata carries where it "came from"
# so results can be traced back to a source, the same way real ingestion
# will tag chunks by filename and section later.
CHUNKS = [
    (
        "If hydraulic pressure falls below 180 bar, inspect the hydraulic "
        "filter for contamination and check the engine-driven pump for "
        "signs of cavitation. Do not attempt to restore pressure by "
        "cycling the pump repeatedly, as this accelerates seal damage.",
        {"source": "hydraulic_manual", "section": "5.4", "category": "hydraulic"},
    ),
    (
        "Cabin altitude above 10000 feet triggers automatic deployment of "
        "passenger oxygen masks. Crew should confirm cabin pressurization "
        "status before initiating an emergency descent.",
        {"source": "pressurization_manual", "section": "3.2", "category": "pressurization"},
    ),
    (
        "Brake temperature exceeding 300 degrees Celsius requires a "
        "cooling period before the next departure. Continued taxi with "
        "overheated brakes risks tire fuse plug activation.",
        {"source": "brake_manual", "section": "7.1", "category": "brakes"},
    ),
    (
        "Engine vibration detected during climb may indicate fan blade "
        "imbalance or a bearing fault. Reduce climb power and monitor "
        "vibration indication before continuing to cruise.",
        {"source": "engine_manual", "section": "2.9", "category": "engine"},
    ),
    (
        "Low actuator pressure at a single flight control surface, with "
        "normal system pressure elsewhere, indicates a restriction "
        "downstream of the manifold rather than a pump fault. Check the "
        "servo valve and shuttle valve for jamming.",
        {"source": "hydraulic_manual", "section": "5.7", "category": "hydraulic"},
    ),
    (
        "Filter differential pressure above 2.5 bar triggers the clogging "
        "indicator. A popped indicator button requires filter element "
        "replacement before the next flight.",
        {"source": "hydraulic_manual", "section": "5.5", "category": "hydraulic"},
    ),
]


def get_embeddings(texts: list[str]) -> list[list[float]]:
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


def connection_string() -> str:
    return (
        f"host=localhost "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ['POSTGRES_DB']} "
        f"user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


def main() -> None:
    texts = [content for content, _metadata in CHUNKS]

    print(f"Embedding {len(texts)} chunk(s) via {EMBEDDING_MODEL}...")
    embeddings = get_embeddings(texts)

    with psycopg.connect(connection_string()) as conn:
        with conn.cursor() as cur:
            inserted_ids = []
            for (content, metadata), embedding in zip(CHUNKS, embeddings):
                cur.execute(
                    """
                    INSERT INTO document_chunks (content, metadata, embedding)
                    VALUES (%s, %s, %s::vector)
                    RETURNING id;
                    """,
                    (
                        content,
                        Jsonb(metadata),
                        vector_to_pgvector(embedding),
                    ),
                )
                inserted_ids.append(cur.fetchone()[0])

    print(f"Inserted {len(inserted_ids)} row(s): ids {inserted_ids}")


if __name__ == "__main__":
    main()