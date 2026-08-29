"""Stage 2 - grounded generation.

    question -> embed -> pgvector search -> best chunks -> Qwen3 -> answer

The one rule this script exists to enforce: Qwen3 must never answer from
its own general knowledge. It only gets to see chunks pulled from
document_chunks, and if pgvector finds nothing close enough, Qwen3 is
never even called - the script just says so.

    python scripts/ask.py "what should I check if hydraulic pressure drops?"
"""

import os
import sys

import httpx
import psycopg


OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
EMBEDDING_MODEL = "embeddinggemma"
LLM_MODEL = "qwen3:4b-instruct"
EMBEDDING_DIMENSIONS = 768

TOP_K = 5

# Cosine distance ceiling. Above this, a chunk is treated as "not actually
# relevant" even if it is the closest thing in the table. Without this,
# LIMIT 5 always hands Qwen3 five chunks - including five wrong ones for a
# question the corpus cannot answer - and Qwen3 will confidently answer
# from them anyway. 0.75 is a starting point, not a measured constant;
# tune it by running ask.py against your own corpus.
MAX_DISTANCE = 0.48

DEFAULT_QUESTION = "What should I check if hydraulic pressure drops below 180 bar?"

SYSTEM_PROMPT = """You are AeroIntel, an aviation technical assistant.

You answer strictly from the numbered CONTEXT blocks given with each \
question. Do not use outside knowledge. Do not fill gaps with what \
aircraft systems normally do.

Rules:
1. Use only the CONTEXT. Cite the block you used, like [1] or [2, 3].
2. If the CONTEXT does not answer the question, say so plainly. Do not \
guess.
3. Quote exact figures and limits from the CONTEXT rather than rounding \
or paraphrasing them.
4. Be concise. No preamble."""

NO_CONTEXT_MESSAGE = (
    "No chunk in the database is close enough to this question "
    "(nothing under distance {ceiling}). Refusing to answer rather than "
    "guessing. Either the relevant document has not been ingested yet, "
    "or the question needs different wording."
)


def get_embedding(text: str) -> list[float]:
    response = httpx.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
        },
        timeout=30.0,
    )
    response.raise_for_status()

    embedding = response.json()["embeddings"][0]

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"expected {EMBEDDING_DIMENSIONS} dimensions, "
            f"got {len(embedding)}"
        )

    return embedding


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


def retrieve_chunks(question: str) -> list[tuple[int, str, dict, float]]:
    """Embed the question, run pgvector search, filter by MAX_DISTANCE.

    Returns [(id, content, metadata, distance), ...] for chunks that
    passed the distance ceiling, closest first. Empty list means: nothing
    in the corpus is relevant enough to answer from.
    """
    query_embedding = get_embedding(question)
    query_vector = vector_to_pgvector(query_embedding)

    with psycopg.connect(connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, metadata, embedding <=> %s::vector AS distance
                FROM document_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_vector, query_vector, TOP_K),
            )
            rows = cur.fetchall()

    return [
        (row_id, content, metadata, float(distance))
        for row_id, content, metadata, distance in rows
        if float(distance) <= MAX_DISTANCE
    ]


def build_prompt(question: str, chunks: list[tuple[int, str, dict, float]]) -> str:
    """Turn retrieved chunks into numbered CONTEXT blocks.

    Block number in the prompt == citation number Qwen3 is told to use,
    so [1] in the answer always maps back to the first block here.
    """
    blocks = []
    for rank, (_id, content, metadata, _distance) in enumerate(chunks, start=1):
        source = metadata.get("source", "unknown")
        blocks.append(f"[{rank}] Source: {source}\n{content}")

    context = "\n\n---\n\n".join(blocks)
    return f"CONTEXT:\n\n{context}\n\n---\n\nQUESTION: {question}\n\nANSWER:"


def generate(prompt: str) -> str:
    response = httpx.post(
        OLLAMA_CHAT_URL,
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=180.0,
    )

    if response.status_code != 200:
        # Ollama's error body (e.g. "model 'qwen3' not found") is far more
        # useful than the bare status code raise_for_status() would give.
        raise RuntimeError(
            f"Ollama chat request failed ({response.status_code}): "
            f"{response.text}"
        )

    return response.json()["message"]["content"].strip()


def main() -> None:
    argv_question = " ".join(sys.argv[1:]).strip()
    question = argv_question or DEFAULT_QUESTION

    print(f"Question: {question}")
    print()

    chunks = retrieve_chunks(question)

    if not chunks:
        print(NO_CONTEXT_MESSAGE.format(ceiling=MAX_DISTANCE))
        return

    print(f"Retrieved {len(chunks)} chunk(s):")
    for rank, (chunk_id, _content, metadata, distance) in enumerate(chunks, start=1):
        source = metadata.get("source", "unknown")
        print(f"  [{rank}] id={chunk_id} distance={distance:.4f} source={source}")
    print()

    prompt = build_prompt(question, chunks)
    answer = generate(prompt)

    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()