from math import sqrt

import httpx


OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "embeddinggemma"

QUERY = "An aircraft engine experienced abnormal vibration during climb."

DOCUMENTS = [
    "The engine began shaking unexpectedly while the aircraft was ascending.",
    "The aircraft powerplant developed severe oscillations while gaining altitude.",
    "The passenger requested vegetarian food during the flight.",
]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a collection of texts using Ollama."""

    response = httpx.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        },
        timeout=30.0,
    )

    response.raise_for_status()

    data = response.json()

    return data["embeddings"]


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Calculate cosine similarity between two vectors."""

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def main() -> None:
    texts = [QUERY, *DOCUMENTS]

    embeddings = get_embeddings(texts)

    query_embedding = embeddings[0]
    document_embeddings = embeddings[1:]

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding dimensions: {len(query_embedding)}")
    print()
    print(f"Query: {QUERY}")
    print()

    results = []

    for document, embedding in zip(
        DOCUMENTS,
        document_embeddings,
    ):
        similarity = cosine_similarity(
            query_embedding,
            embedding,
        )

        results.append(
            (similarity, document)
        )

    results.sort(
        key=lambda result: result[0],
        reverse=True,
    )

    print("Semantic search results:")

    for rank, (score, document) in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. {score:.4f} - {document}"
        )


if __name__ == "__main__":
    main()